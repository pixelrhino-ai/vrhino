#include "neural_graph.h"
#include "vrhino/tensor_util.h"
#ifdef VRHINO_TEST_CUDA
#include "vrhino/backend/cuda_backend.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace vrhino;
namespace ng = vrhino::neural_graph;
namespace {
void check(bool good, const std::string& why) { if (!good) throw std::runtime_error(why); }
ng::Description single(std::vector<ng::TensorSpec> specs, ng::Primitive primitive) {
    ng::Description d; d.schema = ng::schema_v1; d.values = std::move(specs);
    auto output = static_cast<ng::ValueId>(d.values.size()-1);
    for (ng::ValueId i=0;i<output;++i) d.inputs.push_back(i);
    d.nodes={{output,std::move(primitive)}}; d.outputs={output}; return d;
}
void reject(const ng::Description& d,const char* why) {
    bool rejected=false;
    try { (void)ng::admit(d); } catch (const ng::GraphError& e) { rejected=e.phase==ng::Phase::Admit; }
    check(rejected,std::string("Invalid graph accepted: ")+why);
}
void dependency_rejections() {
    auto reject_dependency=[](const ng::Description& d) {
        bool rejected=false;
        try {(void)ng::admit(d);}catch(const ng::GraphError& e){
            rejected=e.phase==ng::Phase::Admit && e.code==ng::Code::InvalidDependency;
        }
        check(rejected,"Unknown/self operand must reject as InvalidDependency");
    };
    // Exercise every new operand slot with an out-of-range ID and a self edge.
    for(ng::ValueId bad : {ng::ValueId{99},ng::ValueId{1}}) {
        reject_dependency(single({{{2,3}},{{3,2}}},ng::Permute{bad,{1,0}}));
        reject_dependency(single({{{2,3}},{{2,2}}},ng::Slice{bad,1,1,3}));
    }
    for(ng::ValueId bad : {ng::ValueId{99},ng::ValueId{2}}) {
        reject_dependency(single({{{2,3}},{{2,1}},{{2,4}}},ng::Concat{{0,bad},1}));
        reject_dependency(single({{{4,3}},{{2},DType::I64},{{2,3}}},ng::Gather{bad,1}));
        reject_dependency(single({{{4,3}},{{2},DType::I64},{{2,3}}},ng::Gather{0,bad}));
    }
    for(ng::ValueId bad : {ng::ValueId{99},ng::ValueId{3}}) {
        for(auto op : {ng::Rope{bad,1,2},ng::Rope{0,bad,2},ng::Rope{0,1,bad}})
            reject_dependency(single({{{1,3,2,4}},{{1,3,1,4}},{{1,3,1,4}},{{1,3,2,4}}},op));
        for(auto op : {ng::Attention{bad,1,2,.5f},ng::Attention{0,bad,2,.5f},ng::Attention{0,1,bad,.5f}})
            reject_dependency(single({{{1,3,2,4}},{{1,2,2,4}},{{1,2,2,4}},{{1,3,2,4}}},op));
    }
    reject(single({{{2,3}},{{3,2}}},ng::Permute{0,{-1,0}}),"negative permutation axis");
    reject(single({{{4,3}},{{2},DType::I8},{{2,3}}},ng::Gather{0,1}),"I8 cannot be gather index");
    reject(single({{{4,3}},{{1,1,1,1,1,1,1,1},DType::I64},{{1}}},ng::Gather{0,1}),"rank8 indices produce forbidden rank9");
    reject(single({{{2}},{{2}},{{2},DType::I64}},ng::Add{0,1}),"integer arithmetic result");
    std::cout<<"DEPENDENCIES\tPASS\tall new operand slots unknown and self IDs; integer result and rank limits\n";
}
void admission() {
    auto p=single({{{2,3}},{{3,2}}},ng::Permute{0,{1,0}});(void)ng::admit(p);
    p.nodes[0].op=ng::Permute{0,{0,0}};reject(p,"duplicate axis");
    p.nodes[0].op=ng::Permute{0,{2,0}};reject(p,"out of range axis");
    p.nodes[0].op=ng::Permute{0,{0}};reject(p,"wrong permutation rank");
    auto s=single({{{2,3}},{{2,2}}},ng::Slice{0,1,1,3});(void)ng::admit(s);
    for(auto op : {ng::Slice{0,1,-1,2},ng::Slice{0,1,2,2},ng::Slice{0,1,0,4},ng::Slice{0,-1,1,3}}) {
        s.nodes[0].op=op;reject(s,"slice noncanonical range");
    }
    auto c=single({{{2,3}},{{2,1}},{{2,4}}},ng::Concat{{0,1},1});(void)ng::admit(c);
    c.nodes[0].op=ng::Concat{{},1};reject(c,"empty concat");
    c.nodes[0].op=ng::Concat{std::vector<ng::ValueId>(17,0),1};reject(c,"unbounded concat");
    c.nodes[0].op=ng::Concat{{0,1},0};reject(c,"concat incompatible dimensions");
    c.nodes[0].op=ng::Concat{{0,1},2};reject(c,"concat axis");
    auto g=single({{{4,3}},{{2},DType::I64},{{2,3}}},ng::Gather{0,1});(void)ng::admit(g);
    auto wrong=g;wrong.values[1].dtype=DType::F32;reject(wrong,"floating gather indices");
    wrong=g;wrong.inputs={0};wrong.parameters={1};reject(wrong,"integer parameter");
    wrong=g;wrong.values[0].shape={1,4,3};reject(wrong,"rank3 gather table");
    wrong=g;wrong.nodes[0].op=ng::Reshape{1,{2,1}};wrong.values[2].shape={2,1};reject(wrong,"integer reshape");
    wrong=g;wrong.nodes[0].op=ng::Add{1,1};wrong.values[2].shape={2};reject(wrong,"integer arithmetic");
    auto r=single({{{1,3,2,4}},{{1,3,1,4}},{{1,3,1,4}},{{1,3,2,4}}},ng::Rope{0,1,2});(void)ng::admit(r);
    wrong=r;wrong.values[1].shape={1,3,2,4};reject(wrong,"unequal rope frequencies");
    wrong=r;wrong.values[0].shape=wrong.values[3].shape={1,3,2,3};reject(wrong,"odd rope width");
    wrong=r;wrong.values[0].shape=wrong.values[3].shape={1,1,2,4};reject(wrong,"rope broadcast expands input");
    auto a=single({{{1,3,2,4}},{{1,2,2,4}},{{1,2,2,4}},{{1,3,2,4}}},ng::Attention{0,1,2,.5f});(void)ng::admit(a);
    for(float scale : {0.f,-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        wrong=a;wrong.nodes[0].op=ng::Attention{0,1,2,scale};reject(wrong,"attention scale");
    }
    wrong=a;wrong.values[2].shape={1,3,2,4};reject(wrong,"attention key/value mismatch");
    wrong=a;wrong.values[0].shape=wrong.values[3].shape={1,3,3,4};reject(wrong,"attention head mismatch");
    wrong=a;wrong.values[0].shape=wrong.values[3].shape={3,2,4};reject(wrong,"attention rank");
    auto silu=single({{{3}},{{3}}},ng::Activate{0,Activation::Silu});(void)ng::admit(silu);
    silu.nodes[0].op=ng::Activate{0,Activation::Gelu};reject(silu,"unadmitted activation");
    std::cout<<"ADMISSION\tPASS\t6 new families, index isolation, activation identity\n";
}
#ifdef VRHINO_TEST_CUDA
std::vector<float> compare(Backend& backend,const ng::Description& d,const std::vector<Tensor>& inputs,
                          const std::vector<float>& expected,const char* label,double atol=0,double rtol=0) {
    auto outputs=ng::evaluate(backend,ng::admit(d),inputs,{});
    check(outputs.size()==1 && outputs[0].device()==DeviceId::accelerator(),"Output must be CUDA");
    const Tensor h=backend.copy_to_host(outputs[0]);
    check(h.numel()==static_cast<int64_t>(expected.size()),"Output size mismatch");
    std::vector<float> actual(h.data_as<float>(),h.data_as<float>()+h.numel());
    double maximum=0,sum=0,relative=0,dot=0,aa=0,bb=0;size_t nonfinite=0;
    for(size_t i=0;i<actual.size();++i) {
        const double x=actual[i],y=expected[i],e=std::abs(x-y);
        nonfinite+=!std::isfinite(x);maximum=std::max(maximum,e);sum+=e;
        relative=std::max(relative,e/std::max(1e-12,std::abs(y)));dot+=x*y;aa+=x*x;bb+=y*y;
        check(std::isfinite(x) && e<=atol+rtol*std::abs(y),std::string(label)+" tolerance mismatch");
    }
    std::cout<<"NUM\t"<<label<<"\t"<<maximum<<"\t"<<sum/actual.size()<<"\t"<<relative<<"\t"
             <<dot/std::sqrt(aa*bb)<<"\t"<<nonfinite<<"\tPASS\n";return actual;
}
void numerical() {
    CudaBackend backend;backend.set_execution_dtype(DType::F32);
    // Non-H3 layout semantics used in LTX and Wan. Handwritten expected element orders.
    auto p=single({{{2,3}},{{3,2}}},ng::Permute{0,{1,0}});
    compare(backend,p,{host_f32({2,3},{0,1,2,3,4,5})},{0,3,1,4,2,5},"non_h3_permute");
    auto s=single({{{2,3}},{{2,2}}},ng::Slice{0,1,1,3});
    compare(backend,s,{host_f32({2,3},{0,1,2,3,4,5})},{1,2,4,5},"non_h3_slice");
    auto c=single({{{2,3}},{{2,1}},{{2,4}}},ng::Concat{{0,1},1});
    compare(backend,c,{host_f32({2,3},{0,1,2,3,4,5}),host_f32({2,1},{6,7})},{0,1,2,6,3,4,5,7},"non_h3_concat");
    // Rank2 T5-style table gather, including repeat and reverse order.
    auto g=single({{{4,3}},{{3},DType::I64},{{3,3}}},ng::Gather{0,1});
    Tensor table=host_f32({4,3},{0,1,2,3,4,5,6,7,8,9,10,11});
    compare(backend,g,{table,host_i64({3},{3,0,3})},{9,10,11,0,1,2,9,10,11},"non_h3_t5_row_gather_i64");
    Tensor i32=Tensor::host({3},DType::I32);int32_t ids[]={3,0,3};std::memcpy(i32.data(),ids,sizeof(ids));
    auto g32=g;g32.values[1].dtype=DType::I32;
    compare(backend,g32,{table,i32},{9,10,11,0,1,2,9,10,11},"non_h3_t5_row_gather_i32");
    bool rejected=false;try{(void)ng::evaluate(backend,ng::admit(g),{table,host_i64({3},{0,4,1})},{});}
    catch(const ng::GraphError& e){rejected=e.phase==ng::Phase::Run;}
    check(rejected,"Out of range gather must fail before kernel");
    compare(backend,g,{table,host_i64({3},{3,0,3})},{9,10,11,0,1,2,9,10,11},"gather_recovery");
    // Generic adjacent RoPE, two heads and token-varying frequencies (Wan semantic).
    std::vector<float> x(24),co(12),si(12),expected(24);
    for(int i=0;i<24;++i)x[i]=(i-11)/16.f;
    for(int t=0;t<3;++t)for(int pair=0;pair<2;++pair)for(int j=0;j<2;++j){
        co[t*4+pair*2+j]=float(std::cos(.13*(t+1)*(pair+1)));
        si[t*4+pair*2+j]=float(std::sin(.13*(t+1)*(pair+1)));
    }
    for(int t=0;t<3;++t)for(int h=0;h<2;++h)for(int j=0;j<4;++j){
        int i=(t*2+h)*4+j;
        expected[i]=float(double(x[i])*co[t*4+j]+double((j%2)?x[i-1]:-x[i+1])*si[t*4+j]);
    }
    auto r=single({{{1,3,2,4}},{{1,3,1,4}},{{1,3,1,4}},{{1,3,2,4}}},ng::Rope{0,1,2});
    compare(backend,r,{host_f32({1,3,2,4},x),host_f32({1,3,1,4},co),host_f32({1,3,1,4},si)},expected,"non_h3_wan_adjacent_rope",2e-7,2e-6);
    // Generic unmasked attention with unequal query/key lengths (LTX cross attention).
    std::vector<float> k(16),v(16);for(int i=0;i<16;++i){k[i]=(i%7-3)/8.f;v[i]=(i%11-5)/8.f;}
    for(int q=0;q<3;++q)for(int h=0;h<2;++h){
        double scores[2]={},maxscore=-1e100;
        for(int key=0;key<2;++key){for(int j=0;j<4;++j)scores[key]+=double(x[(q*2+h)*4+j])*k[(key*2+h)*4+j];scores[key]*=.5;maxscore=std::max(maxscore,scores[key]);}
        double denom=0;for(auto& score:scores){score=std::exp(score-maxscore);denom+=score;}
        for(int j=0;j<4;++j){double sum=0;for(int key=0;key<2;++key)sum+=scores[key]/denom*v[(key*2+h)*4+j];expected[(q*2+h)*4+j]=float(sum);}
    }
    auto a=single({{{1,3,2,4}},{{1,2,2,4}},{{1,2,2,4}},{{1,3,2,4}}},ng::Attention{0,1,2,.5f});
    compare(backend,a,{host_f32({1,3,2,4},x),host_f32({1,2,2,4},k),host_f32({1,2,2,4},v)},expected,"non_h3_ltx_cross_attention",3e-7,3e-6);
    for(size_t i=0;i<x.size();++i)expected[i]=float(double(x[i])/(1+std::exp(-double(x[i]))));
    compare(backend,single({{{24}},{{24}}},ng::Activate{0,Activation::Silu}),{host_f32({24},x)},expected,"non_h3_ltx_silu",2e-7,2e-6);
    std::cout<<"CUDA\tPASS\tall six new families plus Silu; no model forward\n";
}
#endif
}
int main(){try{admission();dependency_rejections();
#ifdef VRHINO_TEST_CUDA
numerical();
#endif
return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
