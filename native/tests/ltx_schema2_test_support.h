#pragma once
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include "../src/architectures/ltx_declaration.h"
#include "vrhino/architecture.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/product/token_grid_geometry.h"
#include "vrhino/tensor_util.h"
#include <array>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace vrhino::ltx_schema2_test {
namespace p = vrhino::product;
namespace fs = std::filesystem;
inline Json set(Json j, const std::string &k, Json v) {
    auto o = j.object();
    o[k] = std::move(v);
    return Json(o);
}
inline Json read(const fs::path &path) {
    std::ifstream f(path);
    require(f.good(), "Missing fixture");
    return Json::parse(std::string(std::istreambuf_iterator<char>(f), {}),
                       JsonParseLimits{1024 * 1024, 32, 10000, 100000, 32768});
}
inline void write(const fs::path &path, const Json &j) {
    std::ofstream f(path);
    f << j.serialize();
    require(f.good(), "Cannot save fixture");
}
inline Json geometry() {
    return Json::parse(
        R"({"kind":"token_grid_thw.v1","spatial_scale":32,"temporal_scale":8,"temporal_origin":1,"channels":128,"flatten_order":"THW_W_FASTEST"})");
}
inline Json programs(const SamplingProgram &p) {
    auto j = Json::parse(
        R"({"schema":"vrhino.programs.v2","required_capabilities":["execution.per_step.v1","sampling.flow_euler_cfg.v1"],"execution":{"kind":"per_step","instance_ids":[]},"sampling":{"prediction":"flow","solver":"flow_euler","maximum_order":1,"schedule":"flow_sigma","model_timestep":{"source":"sigma","dtype":"float32","shape":[1,1]},"branch_order":"unconditional_then_conditional","transitions":[]}})");
    Json::Array ids, steps;
    for (int i = 0; i < p.steps; ++i) {
        ids.emplace_back(int64_t(0));
        steps.emplace_back(
            Json::Object{{"sigma", Json(double(p.sigmas.at(i)))},
                         {"next_sigma", Json(double(p.sigmas.at(i + 1)))},
                         {"guidance_scale", Json(double(p.guidance_coefficients.at(1)))}});
    }
    j = set(j, "execution", set(j.at("execution"), "instance_ids", Json(ids)));
    return set(j, "sampling", set(j.at("sampling"), "transitions", Json(steps)));
}
struct Source : p::TensorSource {
    std::map<std::string, p::SourceTensorDescriptor> values;
    const std::map<std::string, p::SourceTensorDescriptor> &tensors() const noexcept override {
        return values;
    }
    void read_tensor(const p::SourceTensorDescriptor &d, uint64_t offset, void *buffer,
                     size_t bytes) const override {
        require(offset % dtype_size(d.dtype) == 0 && bytes % dtype_size(d.dtype) == 0,
                "Fixture alignment");
        // Deterministic bounded nonzero data, including norm/stats/modulation.
        uint32_t hash = 2166136261u;
        for (unsigned char c : d.name)
            hash = (hash ^ c) * 16777619u;
        const size_t count = bytes / dtype_size(d.dtype);
        const uint64_t start = offset / dtype_size(d.dtype);
        const float base = (d.name.find("norm.weight") != std::string::npos ||
                            d.name.find("std-of-means") != std::string::npos)
                               ? 1.0f
                               : 0.0f;
        const float scale =
            d.name.find("scale_shift_table") != std::string::npos ? 0.015625f : 0.0001220703125f;
        for (size_t i = 0; i < count; ++i) {
            const float value = base + float(int((start + i + hash) % 17) - 8) * scale;
            if (d.dtype == DType::F32)
                static_cast<float *>(buffer)[i] = value;
            else {
                require(d.dtype == DType::BF16, "Fixture dtype");
                static_cast<uint16_t *>(buffer)[i] = std::bit_cast<uint32_t>(value) >> 16;
            }
        }
    }
};
inline void add(Source &source, std::vector<p::TensorMapping> &mappings, const std::string &name,
                const ArchitectureParameterSlot &s) {
    p::SourceTensorDescriptor d;
    d.name = name;
    d.dtype = s.dtype;
    d.source_dtype = dtype_name(s.dtype);
    d.shape = s.shape;
    d.byte_length = uint64_t(shape_numel(s.shape)) * dtype_size(s.dtype);
    source.values.emplace(name, d);
    mappings.emplace_back(name, s.dtype, s.shape, "identity_bytes", name, s.dtype, s.shape,
                          "parameter", "weight");
}
inline std::shared_ptr<VrmModel> marker(const fs::path &path, const Json &graph, const Json &meta,
                                        const std::string &architecture) {
    Source source;
    std::vector<p::TensorMapping> maps;
    add(source, maps, "marker", {"marker", {1}, DType::F32});
    fs::remove(path);
    p::write_vrm_streaming(path, "dit-flow", architecture, meta, graph, source, maps);
    return std::make_shared<VrmModel>(path.string());
}
inline std::shared_ptr<VrmModel> legacy_marker(const fs::path &dir) {
    return marker(
        dir / "legacy-marker.vrm",
        Json::parse(
            R"({"schema_version":1,"architecture_graph":{"schema_version":1,"config":{"rope_coordinate_scale":[0.32,32,32]},"runtime_tensor_bindings":{"marker":"marker"}},"component_graphs":[{"schema_version":1,"runtime_tensor_bindings":{"marker":"marker"}}]})"),
        Json::parse(R"({"architecture":"ltx_v0_9_1"})"), "ltx_v0_9_1");
}
inline SamplingProgram legacy_schedule(Architecture &a, int64_t tokens, int steps = 3) {
    TensorBundle input{{"seed", scalar_i64(11)},
                       {"coordinates", Tensor::host({1, 3, tokens}, DType::F32)},
                       {"sampling_steps", scalar_i64(steps)},
                       {"guidance_scale", scalar_f32(3)}};
    return a.create_program(input);
}
inline Json catalog(const Json &declaration, const Json::Object &parameters) {
    return Json(Json::Object{
        {"schema_version", Json(int64_t(2))},
        {"required_capabilities", Json(Json::Array{Json(std::string("binding_catalog.v1"))})},
        {"graphs", Json(Json::Array{Json(
                       Json::Object{{"id", Json(int64_t(0))}, {"declaration", declaration}})})},
        {"bindings", Json(Json::Array{Json(Json::Object{{"id", Json(int64_t(0))},
                                                        {"graph", Json(int64_t(0))},
                                                        {"parameters", Json(parameters)}})})},
        {"instances", Json(Json::Array{Json(Json::Object{{"id", Json(int64_t(0))},
                                                         {"graph", Json(int64_t(0))},
                                                         {"binding", Json(int64_t(0))}})})}});
}
inline void exact(const Tensor &a, const Tensor &b) {
    require(a.shape() == b.shape() && a.dtype() == b.dtype() && a.bytes() == b.bytes() &&
                std::memcmp(a.data(), b.data(), a.bytes()) == 0,
            "Tensor byte mismatch");
}
inline size_t fds() {
#ifdef _WIN32
    DWORD count = 0;
    require(GetProcessHandleCount(GetCurrentProcess(), &count), "Cannot count process handles");
    return count;
#elif defined(__linux__)
    return std::distance(fs::directory_iterator("/proc/self/fd"), fs::directory_iterator{});
#else
    return std::distance(fs::directory_iterator("/dev/fd"), fs::directory_iterator{});
#endif
}
inline int negatives = 0;
template <class F> void reject(const std::string &name, F f) {
    auto before = fds();
    bool failed = false;
    try {
        f();
    } catch (const std::exception &e) {
        failed = true;
        std::cout << "REJECT " << name << ": " << e.what() << '\n';
    }
    require(failed, "Unexpected acceptance: " + name);
    require(fds() == before, "File descriptor leak: " + name);
    ++negatives;
}
} // namespace vrhino::ltx_schema2_test
