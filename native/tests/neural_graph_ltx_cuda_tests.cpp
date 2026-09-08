#include "neural_graph.h"
#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"

#include <cuda_runtime_api.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <type_traits>

namespace {
using namespace vrhino;
namespace ng = vrhino::neural_graph;
using Floats = std::vector<float>;
size_t sync_calls = 0, minimum_free = SIZE_MAX;
void check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void cuda_check(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }

struct Fixture {
    std::vector<std::vector<int64_t>> shapes;
    std::vector<Floats> v;
    explicit Fixture(int m) {
        shapes = {{1,3,8},{1,m,8},{1,m,8},{1,m,8},{32,8},{32},{8,32},{8},{1},
                  {1,3,8},{1,m,8},{1,3,8},{1,3,8},{1,3,32},{1,3,32},
                  {1,3,8},{1,3,8},{1,3,8}};
        for (const auto& shape : shapes) v.emplace_back(shape_numel(shape));
        for (int j = 0; j < 8; ++j)
            for (size_t i = 0; i < v[j].size(); ++i)
                v[j][i] = (static_cast<int>((i*17+j*11+3)%41)-20)/32.0f * (j >= 4 ? .25f : 1.0f);
        v[8][0] = 1.0f;
        // Independent scalar mathematics, no Backend, architecture, or graph.
        for (int row = 0; row < 3; ++row) {
            double square = 0;
            for (int k = 0; k < 8; ++k) square += double(v[0][row*8+k])*v[0][row*8+k];
            const double inverse = 1.0/std::sqrt(square/8 + double(1e-6f));
            for (int k = 0; k < 8; ++k) v[9][row*8+k] = float(v[0][row*8+k]*inverse);
        }
        for (size_t i = 0; i < v[10].size(); ++i) v[10][i] = float(1.0 + v[2][i]);
        for (int i = 0; i < 24; ++i) {
            int j = m == 1 ? i%8 : i;
            v[11][i] = float(double(v[9][i])*v[10][j]);
            v[12][i] = float(double(v[11][i])+v[1][j]);
        }
        auto linear = [&](int x, int w, int bias, int out, int k, int n) {
            for (int row = 0; row < 3; ++row) for (int col = 0; col < n; ++col) {
                double sum = 0;
                for (int inner = 0; inner < k; ++inner)
                    sum += double(v[x][row*k+inner])*v[w][col*k+inner];
                // Backend GEMM followed by a separate bias addition.
                v[out][row*n+col] = float(double(float(sum))+v[bias][col]);
            }
        };
        linear(12,4,5,13,8,32);
        for (size_t i = 0; i < v[14].size(); ++i) {
            double x = v[13][i];
            v[14][i] = float(.5*x*(1+std::tanh(std::sqrt(2.0/std::acos(-1.0))*(x+.044715*x*x*x))));
        }
        linear(14,6,7,15,32,8);
        for (int i = 0; i < 24; ++i) {
            v[16][i] = float(double(v[15][i])*v[3][m == 1 ? i%8 : i]);
            v[17][i] = float(double(v[0][i])+v[16][i]);
        }
    }
    void dump(int m) const {
        for (size_t id = 0; id < v.size(); ++id) {
            std::cout << "FIXTURE\t" << m << '\t' << id << '\t';
            for (auto dim : shapes[id]) std::cout << dim << ',';
            std::cout << '\t';
            for (float x : v[id]) {
                uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
                std::cout << std::hex << std::setw(8) << std::setfill('0') << bits << ',';
            }
            std::cout << std::dec << '\n';
        }
    }
};

ng::Description description(int m) {
    ng::Description d;
    d.schema = ng::schema_v1;
    d.values = {{{1,3,8}},{{1,m,8}},{{1,m,8}},{{1,m,8}},{{32,8}},{{32}},{{8,32}},{{8}},{{1}},
                {{1,3,8}},{{1,m,8}},{{1,3,8}},{{1,3,8}},{{1,3,32}},{{1,3,32}},
                {{1,3,8}},{{1,3,8}},{{1,3,8}}};
    d.inputs = {0,1,2,3}; d.parameters = {4,5,6,7}; d.constants = {{8,1.0f}};
    d.nodes = {{9,ng::RmsNorm{0,{},2,1e-6f}}, {10,ng::Add{8,2}}, {11,ng::Mul{9,10}},
               {12,ng::Add{11,1}}, {13,ng::Linear{12,4,5}},
               {14,ng::Activate{13,Activation::GeluTanh}}, {15,ng::Linear{14,6,7}},
               {16,ng::Mul{15,3}}, {17,ng::Add{0,16}}};
    d.outputs = {9,10,11,12,13,14,15,16,17};
    return d;
}
ng::Description reindex(ng::Description d) {
    const auto original = d.values;
    auto remap = [](ng::ValueId id) { return ng::ValueId{17}-id; };
    for (ng::ValueId i = 0; i < 18; ++i) d.values[remap(i)] = original[i];
    for (auto& id : d.inputs) id = remap(id);
    for (auto& id : d.parameters) id = remap(id);
    for (auto& c : d.constants) c.value = remap(c.value);
    for (auto& n : d.nodes) {
        n.result = remap(n.result);
        std::visit([&](auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T,ng::Add> || std::is_same_v<T,ng::Mul>) {
                op.a = remap(op.a); op.b = remap(op.b);
            } else if constexpr (std::is_same_v<T,ng::Linear> || std::is_same_v<T,ng::RmsNorm> ||
                                 std::is_same_v<T,ng::Activate> || std::is_same_v<T,ng::Reshape>) {
                op.x = remap(op.x);
                if constexpr (std::is_same_v<T,ng::Linear>) {
                    op.weight = remap(op.weight); if (op.bias) op.bias = remap(*op.bias);
                } else if constexpr (std::is_same_v<T,ng::RmsNorm>) {
                    if (op.weight) op.weight = remap(*op.weight);
                }
            } else if constexpr (!std::is_same_v<T,std::monostate>)
                check(false, "Unexpected primitive in fixed reindex fixture");
        }, n.op);
    }
    for (auto& id : d.outputs) id = remap(id);
    return d;
}

// Exact bounded extraction of ltx.cpp:118-130. Production helper calls, source
// defaults and F32 policy preserved; no graph construction/evaluation here.
std::map<int,Tensor> imperative_ltx(Backend& b, const std::vector<Tensor>& x,
                                   const std::vector<Tensor>& p) {
    const auto policy = PrecisionPolicy::fp32();
    std::map<int,Tensor> t;
    t[9] = b.rms_norm(x[0], nullptr, 1e-6f);
    t[12] = modulate(b, policy, t[9], x[1], x[2]);
    t[13] = b.linear(t[12], p[0], &p[1]);
    t[14] = b.activation(t[13], Activation::GeluTanh);
    t[15] = b.linear(t[14], p[2], &p[3]);
    t[17] = gated_residual(b, policy, x[0], t[15], x[3]);
    b.synchronize();
    return t;
}
Floats read(Backend& b, const Tensor& t, const std::vector<int64_t>& shape) {
    check(t.shape() == shape && t.dtype() == DType::F32 && t.logical_dtype() == DType::F32 &&
          !t.is_quantized(), "output metadata mismatch");
    cudaPointerAttributes attr{}; cuda_check(cudaPointerGetAttributes(&attr,t.data()));
    check(attr.type == cudaMemoryTypeDevice && t.device() == DeviceId::accelerator(), "not real CUDA output");
    auto host = b.copy_to_host(t);
    return {host.data_as<float>(),host.data_as<float>()+host.numel()};
}
void compare(const Floats& actual, const Floats& expected, bool exact,
             int m, int id, const char* path, bool report = true) {
    check(actual.size() == expected.size(), "comparison extent mismatch");
    double maximum=0, sum=0, relative=0, dot=0, aa=0, bb=0;
    size_t nonfinite=0;
    bool accepted=true;
    for (size_t i=0; i<actual.size(); ++i) {
        nonfinite += !std::isfinite(actual[i]) || !std::isfinite(expected[i]);
        double a=actual[i], e=expected[i], err=std::abs(a-e);
        maximum=std::max(maximum,err); sum+=err;
        if (e != 0) relative=std::max(relative,err/std::abs(e));
        dot+=a*e; aa+=a*a; bb+=e*e;
        accepted &= err <= 2e-5+2e-5*std::abs(e);
    }
    bool bits=std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)) == 0;
    if (report) std::cout << "NUM\t" << m << '\t' << id << '\t' << path << '\t'
        << actual.size() << '\t' << maximum << '\t' << sum/actual.size() << '\t' << relative
        << '\t' << dot/std::sqrt(aa*bb) << '\t' << nonfinite << '\t' << bits << '\n';
    check(nonfinite == 0 && (exact ? bits : accepted), "numerical qualification failed");
}
void dispatch(Backend& b) {
    std::map<std::string,uint64_t> counts;
    for (const auto& [key,stat] : b.profile_stats()) {
        std::cout << "DISPATCH\t" << key << '\t' << stat.calls << '\n';
        counts[key.substr(0,key.find('|'))] += stat.calls;
    }
    check(counts["norm.rms"] == 1 && counts["elementwise.add"] == 3 &&
          counts["elementwise.mul"] == 2 && counts["linear"] == 2 && counts["activation"] == 1,
          "unexpected production CUDA dispatch");
}
void resource(const char* phase, size_t backend_peak=0) {
    size_t free=0,total=0; cuda_check(cudaMemGetInfo(&free,&total)); minimum_free=std::min(minimum_free,free);
    rusage r{}; check(getrusage(RUSAGE_SELF,&r) == 0,"RSS query failed");
    std::cout << "RESOURCE\t" << phase << '\t' << free << '\t' << total-free << '\t'
              << total-minimum_free << '\t' << backend_peak << '\t' << r.ru_maxrss << '\n';
}
void qualify(Backend& b, int m) {
    Fixture f(m);
    std::vector<Tensor> inputs,parameters;
    for (int j=0;j<8;++j) {
        Tensor t=b.copy_to_device(host_f32(f.shapes[j],f.v[j]),DType::F32);
        (j<4 ? inputs : parameters).push_back(t);
    }
    const auto g=ng::admit(description(m));
    std::map<int,Floats> reference_first;
    std::vector<Floats> graph_first;
    std::vector<Tensor> held;
    for (int repeat=0;repeat<20;++repeat) {
        const auto ref=imperative_ltx(b,inputs,parameters);
        b.enable_profiling(repeat == 0);
        size_t before=sync_calls;
        auto out=ng::evaluate(b,g,inputs,parameters);
        check(sync_calls == before+1,"graph must synchronize once before capture");
        cuda_check(cudaStreamQuery(nullptr));
        if (repeat == 0) { dispatch(b); held=out; }
        b.enable_profiling(false);
        for (int id=9;id<=17;++id) {
            auto actual=read(b,out[id-9],f.shapes[id]);
            compare(actual,f.v[id],false,m,id,"independent_host",repeat == 0);
            if (ref.count(id)) {
                auto expected=read(b,ref.at(id),f.shapes[id]);
                compare(actual,expected,true,m,id,"imperative",repeat == 0);
                if (repeat == 0) reference_first[id]=expected;
                else compare(expected,reference_first.at(id),true,m,id,"reference_repeat",false);
            }
            if (repeat == 0) graph_first.push_back(actual);
            else compare(actual,graph_first[id-9],true,m,id,"graph_repeat",false);
        }
        resource("repeat",b.peak_device_bytes());
    }
    auto renamed=ng::evaluate(b,ng::admit(reindex(description(m))),inputs,parameters);
    for (int id=9;id<=17;++id) {
        compare(read(b,renamed[id-9],f.shapes[id]),graph_first[id-9],true,m,id,"reindexed");
        compare(read(b,held[id-9],f.shapes[id]),graph_first[id-9],true,m,id,"held_after_repeats",false);
    }
    auto final=description(m); final.outputs={17};
    auto output=ng::evaluate(b,ng::admit(final),inputs,parameters);
    compare(read(b,output[0],f.shapes[17]),reference_first.at(17),true,m,17,"final_only");
    // Inputs and parameters remain unmodified by either path.
    for (int j=0;j<8;++j)
        compare(read(b,j<4 ? inputs[j] : parameters[j-4],f.shapes[j]),f.v[j],true,m,j,"input_immutable",false);
    std::cout << "PASS\tM=" << m << "\t20_graph_and_20_reference_bitwise_repeats"
              << "\treindexed\tfinal_only\tretained_outputs\timmutable_bindings\n";
}
} // namespace
extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    ++sync_calls; return __real_cudaDeviceSynchronize();
}
int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(17);
        if (argc == 2 && std::string(argv[1]) == "--dump-host-fixture") {
            Fixture(1).dump(1); Fixture(3).dump(3); return 0;
        }
        check(argc == 1,"unsupported test argument");
        auto start=std::chrono::steady_clock::now();
        resource("before_backend");
        {
            CudaBackend cuda; Backend& b=cuda; b.set_execution_dtype(DType::F32);
            std::cout << "BACKEND\t" << b.name() << '\n';
            resource("backend_initialized");
            qualify(b,1); qualify(b,3);
            resource("before_destroy",b.peak_device_bytes());
        }
        resource("after_destroy");
        std::cout << "WALL_SECONDS\t" << std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "E3 FAIL: " << e.what() << '\n'; return 1;
    }
}
