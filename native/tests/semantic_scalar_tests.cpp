#include "vrhino/backend/cuda_backend.h"
#include "vrhino/precision.h"
#include "vrhino/json.h"
#include "vrhino/tensor_util.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <cstring>
using namespace vrhino;
template<class F> void reject(F fn) {bool rejected=false;try {fn();}catch(const Error&) {rejected=true;}require(rejected,"Expected fail closed");}
std::string load(const char* path){std::ifstream f(path);require(bool(f),"Missing policy");return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char** argv){try {
 require(argc==3,"usage: semantic-scalar-tests LEGACY POLICY_V2");
 auto legacy=PrecisionPolicy::from_json(Json::parse(load(argv[1]))),p=PrecisionPolicy::from_json(Json::parse(load(argv[2])));
 const auto role=PrecisionScalarRole::SolverCoefficient;
 require(!legacy.has_scalar_role(role) && p.has_scalar_role(role),"Opt-in boundary");
 reject([&]{p.with_scalar_roles({role});});reject([&]{p.has_scalar_role(static_cast<PrecisionScalarRole>(99));});
 const auto text=load(argv[2]);
 for(const auto& [before,after]:std::vector<std::pair<std::string,std::string>>{{"semantic-scalar.v1","semantic-scalar.v99"},{"GUIDANCE_COEFFICIENT","UNKNOWN"},{"TIMESTEP_SCALE","SIGMA"},{"vrhino.precision.policy.v2","vrhino.precision.policy.v1"}}){auto bad=text;bad.replace(bad.find(before),before.size(),after);reject([&]{PrecisionPolicy::from_json(Json::parse(bad));});}
 CudaBackend b;b.set_execution_dtype(DType::BF16);
 auto x=b.copy_to_device(host_f32({4},{10,1,-3,0.5f}),DType::BF16);
 auto mul=[&](const PrecisionPolicy& policy,const Tensor& c){return precision_scalar_binary(b,policy,role,ScalarBinaryOperation::Multiply,x,c);};
 auto c=scalar_f32(.877877295f);
 auto a=b.copy_to_host(mul(p,c));auto d=b.copy_to_host(mul(p,b.copy_to_device(c,DType::F32)));
 require(a.dtype()==DType::BF16 && std::memcmp(a.data(),d.data(),a.bytes())==0,"Placement changes semantic coefficient");
 auto l=b.copy_to_host(mul(legacy,c)),raw=b.copy_to_host(b.mul(x,c));
 require(std::memcmp(l.data(),raw.data(),l.bytes())==0,"Legacy scalar changed");
 require(std::memcmp(a.data(),l.data(),a.bytes())!=0,"Counterexample failed to distinguish contracts");
 reject([&]{mul(p,scalar_f32(std::numeric_limits<float>::quiet_NaN()));});
 reject([&]{mul(p,scalar_f32(std::numeric_limits<float>::infinity()));});
 reject([&]{mul(p,host_f32({2},{1,2}));});reject([&]{mul(p,scalar_i64(1));});
 reject([&]{mul(p,b.copy_to_device(c,DType::BF16));});
 reject([&]{precision_scalar_binary(b,p,PrecisionScalarRole::TimestepScale,ScalarBinaryOperation::Multiply,x,c);});
 reject([&]{precision_scalar_binary(b,p,role,ScalarBinaryOperation::Divide,x,scalar_f32(0));});
 auto data=b.copy_to_device(c,DType::BF16);
 auto data_out=precision_scalar_binary(b,p,PrecisionScalarRole::DataOperand,ScalarBinaryOperation::Multiply,x,data);
 auto data_host=b.copy_to_host(data_out);require(std::memcmp(data_host.data(),raw.data(),raw.bytes())==0,"Data operand implicitly promoted");
 auto control=precision_scalar_binary(b,p,PrecisionScalarRole::TimestepScale,ScalarBinaryOperation::Multiply,host_f32({2,1},{.877877295f,.865800679f}),scalar_f32(1000));
 require(control.dtype()==DType::F32,"Control coordinate narrowed");
 auto actual=b.copy_to_host(control);require(actual.data_as<float>()[0]==.877877295f*1000,"Control value changed");
 // Roles are independent: opting in guidance cannot silently change solver.
 auto guidance=legacy.with_scalar_roles({PrecisionScalarRole::GuidanceCoefficient});
 auto isolated=b.copy_to_host(mul(guidance,c));require(std::memcmp(isolated.data(),raw.data(),raw.bytes())==0,"Role isolation failed");
 b.synchronize();std::cout<<"semantic_scalar_tests=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
