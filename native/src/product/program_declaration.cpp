#include "vrhino/product/program_declaration.h"
#include "vrhino/error.h"
#include <cmath>
#include <set>
#include <limits>
#include <algorithm>

namespace vrhino::product {
namespace {
void keys(const Json& j, std::initializer_list<const char*> expected) {
    require(j.object().size()==expected.size(), "Program declaration field set mismatch");
    for (auto k:expected) require(j.find(k), "Missing program field");
}
}
DeclaredPrograms admit_program_declaration(const Json& j, const PackageDeclaration& package,
                                           const std::vector<std::string>& supported) {
    keys(j,{"schema","required_capabilities","execution","sampling"});
    require(j.at("schema").string()=="vrhino.programs.v1", "Unsupported program schema");
    std::set<std::string> caps;
    for (const auto& cap:j.at("required_capabilities").array()) {
        require(std::find(supported.begin(),supported.end(),cap.string())!=supported.end(),
                "Unsupported required program capability");
        require(caps.insert(cap.string()).second,"Duplicate program capability");
    }
    require(caps==std::set<std::string>{"execution.per_step.v1","sampling.flow_sigma_cfg.v1"},
            "Incomplete program capability declaration");
    const auto& e=j.at("execution"); keys(e,{"kind","instance_ids"});
    require(e.at("kind").string()=="per_step","Unsupported execution declaration");
    std::vector<ComponentInstanceID> ids;
    const Json* interface=nullptr;
    for (const auto& id:e.at("instance_ids").array()) {
        auto n=id.integer(); require(n>=0 && n<=std::numeric_limits<uint32_t>::max(),"Invalid instance ID");
        auto it=package.instances().find(static_cast<uint32_t>(n));
        require(it!=package.instances().end(),"Unknown execution instance reference");
        const auto& g=package.graphs().at(it->second.graph);
        if (interface) require(interface->serialize()==g.serialize(),"Incompatible selectable graph interface");
        interface=&g; ids.push_back({static_cast<uint32_t>(n)});
    }
    const auto& s=j.at("sampling");
    keys(s,{"prediction","solver","maximum_order","schedule","branch_order","transitions"});
    require(s.at("prediction").string()=="flow" &&
            s.at("solver").string()=="multistep_predictor_corrector" &&
            s.at("maximum_order").integer()==2 && s.at("schedule").string()=="flow_sigma" &&
            s.at("branch_order").string()=="unconditional_then_conditional","Unsupported sampling semantics");
    require(!ids.empty() && ids.size()<=10000 && ids.size()==s.at("transitions").array().size(),
            "Execution/sampling step table length mismatch");
    std::vector<FlowScheduleTransition> transitions;
    std::vector<GuidanceParameters> guidance;
    for (const auto& t:s.at("transitions").array()) {
        keys(t,{"model_timestep","sigma","next_sigma","guidance_scale"});
        const auto time=t.at("model_timestep").integer();
        require(time>=0,"Invalid model timestep");
        Tensor tensor=Tensor::host({},DType::I64); *tensor.data_as<int64_t>()=time;
        for (auto key:{"sigma","next_sigma","guidance_scale"})
            require(std::isfinite(t.at(key).number()),"Nonfinite sampling parameter");
        require(t.at("guidance_scale").number()>=0,"Invalid CFG scale");
        transitions.push_back({tensor,static_cast<float>(t.at("sigma").number()),
                               static_cast<float>(t.at("next_sigma").number())});
        guidance.push_back(GuidanceParameters::cfg(static_cast<float>(t.at("guidance_scale").number())));
    }
    SamplingProgram sampling; sampling.steps=static_cast<int>(ids.size()); sampling.guidance_mode=GuidanceMode::CFG;
    sampling.contract=SamplingContract{{PredictionSemantic::Flow},
        {SolverSemantic::MultistepPredictorCorrector,2},ScheduleContract::flow_sigma(std::move(transitions))};
    sampling.guidance_schedule=GuidanceSchedule::per_step(std::move(guidance));
    sampling.guidance_schedule->validate(ids.size());
    return {ExecutionProgram::per_step(std::move(ids)),std::move(sampling)};
}
} // namespace vrhino::product
