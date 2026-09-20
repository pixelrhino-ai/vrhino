#include "vrhino/product/component_preparation.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/tensor_util.h"
#include "vrhino/error.h"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <limits>
#include <iostream>

using namespace vrhino;
namespace p=vrhino::product;
int main(int argc,char** argv) {try {
    require(argc==2,"usage: product-component-tests OUTPUT");
    const std::filesystem::path dir=argv[1];std::filesystem::create_directories(dir);
    const auto file=dir/"weights.safetensors";
    std::string header=R"({"embedding":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]},"norm":{"dtype":"F32","shape":[4],"data_offsets":[48,64]}})";
    while(header.size()%8)header+=' ';
    const std::vector<float> values{1,2,3,4, -1,2,-3,4, 4,3,2,1, 1,1,1,1};
    {std::ofstream f(file,std::ios::binary);uint64_t size=header.size();f.write(reinterpret_cast<const char*>(&size),8);f<<header;f.write(reinterpret_cast<const char*>(values.data()),64);}
    p::PreparedTextProductRequest request;request.owner=std::make_shared<p::AdmittedLocalProduct>();
    request.owner->conditioning=std::make_unique<p::SafeTensorAsset>(p::SafeTensorAsset::single(file));
    const auto graph=Json::parse(R"({"schema_version":1,"kind":"pre_norm_transformer","embedding":{"weight":"embedding"},"blocks":[],"final_norm":{"kind":"rms_norm","weight":"norm","eps":0.000001},"output_trim_to_mask":true})");
    request.runtime_inputs={{"seed",scalar_i64(7)}};
    request.conditioning.push_back({"positive",graph,host_i64({1,3},{0,1,2}),host_bool({1,3},{1,1,0}),{1,2,4}});
    request.conditioning.push_back({"negative",graph,host_i64({1,3},{1,0,2}),host_bool({1,3},{1,0,0}),{1,1,4}});
    std::weak_ptr<p::AdmittedLocalProduct> owner=request.owner;
    TensorBundle output;
    {
        CudaBackend backend;backend.set_execution_dtype(DType::F32);
        output=p::execute_text_conditioning(backend,request);
        require(output.at("positive").shape()==std::vector<int64_t>({1,2,4}),"Positive shape");
        require(output.at("negative").shape()==std::vector<int64_t>({1,1,4}),"Negative shape");
        for(int64_t i=0;i<8;++i) {
            const float expected=values[i]/std::sqrt(7.5f+1e-6f);
            require(std::abs(output.at("positive").data_as<float>()[i]-expected)<=2e-5f,"Known RMS component output");
        }
        request.owner.reset();require(!owner.expired() && backend.resource_owner_count()==1,"Missing backend source lease");
    }
    require(owner.expired(),"Source lease not released at teardown");
    p::require_finite_component_output(output.at("positive"));
    const auto reject=[](auto fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}require(rejected,"Invalid component output admitted");};
    reject([&]{p::require_finite_component_output(host_f32({1},{std::numeric_limits<float>::quiet_NaN()}));});
    reject([&]{p::require_finite_component_output(host_f32({1},{std::numeric_limits<float>::infinity()}));});
    reject([&]{p::require_finite_component_output(host_i64({1},{1}));});
    reject([&]{CudaBackend b;p::execute_text_conditioning(b,request);});
    std::cout<<"PASS generic native conditioning execution; analytic RMS output; mask trimming; separate prompt bindings; host results after device teardown; owner leases; NaN/Inf/type/missing-owner fail closed; no denoiser\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
