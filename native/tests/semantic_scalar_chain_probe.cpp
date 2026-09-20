// Replays fixed branch predictions through the real continuous SamplingRuntime.
// It isolates solver semantics; this is not a denoiser/reference qualification.
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/json.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"
#include <fstream>
#include <iostream>
using namespace vrhino;
Json load(const char* path){std::ifstream f(path);require(bool(f),"Missing declaration");return Json::parse(std::string(std::istreambuf_iterator<char>(f),{}));}
class Replay final:public Denoiser {
public:
 Replay(Backend& b,const TensorBundle& source):b_(b),source_(source){}
 std::vector<Tensor> evaluate(const Tensor&,const Tensor&) override {
  const auto prefix="step."+std::to_string(step_++)+".prediction.";
  const auto& a=source_.at(prefix+"0");const auto& c=source_.at(prefix+"1");
  return {b_.copy_to_device(a,a.dtype()),b_.copy_to_device(c,c.dtype())};
 }
private:Backend& b_;const TensorBundle& source_;size_t step_=0;
};
int main(int argc,char** argv){try {
 require(argc==5,"usage: semantic-scalar-chain-probe INPUT CONTRACT POLICY OUTPUT");
 auto source=read_bundle(argv[1]);const auto contract=load(argv[2]);const auto policy=PrecisionPolicy::from_json(load(argv[3]));
 CudaBackend b;b.set_execution_dtype(policy.requested_dtype());Replay denoiser(b,source);
 SamplingProgram p;p.steps=static_cast<int>(contract.at("steps").integer());p.seed=static_cast<uint64_t>(contract.at("seed").integer());p.latent_shape=source.at("initial_noise").shape();p.guidance_mode=GuidanceMode::CFG;
 std::vector<FlowScheduleTransition> schedule;std::vector<GuidanceParameters> guidance;
 for(const auto& row:contract.at("transitions").array()) {
  schedule.push_back({scalar_i64(row.at("model_timestep").integer()),static_cast<float>(row.at("sigma").number()),static_cast<float>(row.at("next_sigma").number())});
  guidance.push_back(GuidanceParameters::cfg(static_cast<float>(row.at("guidance_scale").number())));
 }
 p.contract=SamplingContract{{PredictionSemantic::Flow},{SolverSemantic::MultistepPredictorCorrector,2},ScheduleContract::flow_sigma(std::move(schedule))};
 p.guidance_schedule=GuidanceSchedule::per_step(std::move(guidance));
 SamplingRuntime runtime(b,policy);runtime.set_solver_trace_enabled(true);
 auto result=runtime.run_with_initial_state(denoiser,p,source.at("initial_noise"));
 result.trace["final_latent"]=b.copy_to_host(result.final_latent);write_bundle(argv[4],result.trace);
 std::cout<<"semantic_scalar_chain_probe=PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
