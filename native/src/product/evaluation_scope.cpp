#include "vrhino/product/evaluation.h"
#include "vrhino/error.h"
#include <set>
#include <cmath>

namespace vrhino::product {
EvaluationScope EvaluationScope::parse(const Json& j) {
    const std::set<std::string> keys{"schema","precision","precision_policy_artifact",
        "mode","max_steps","device_budget_bytes","host_budget_bytes","max_output_bytes",
        "max_latent_elements","max_video_elements","timeout_seconds","fps","media_range","weight_cache_budget_bytes"};
    const bool cache_declared=j.object().contains("weight_cache_budget_bytes");
    require(j.object().size()==keys.size()-(cache_declared?0:1),"Invalid evaluation declaration fields");
    for(const auto& [k,v]:j.object()) { (void)v; require(keys.contains(k),"Unknown evaluation field"); }
    require(j.at("schema").string()=="vrhino.product-evaluation.v1", "Unsupported evaluation schema");
    require(j.at("precision").string()=="bf16", "Evaluation v1 requires declared BF16 policy");
    EvaluationScope s; s.precision_artifact=j.at("precision_policy_artifact").string();
    require(!s.precision_artifact.empty(),"Missing evaluation precision artifact");
    const auto mode=j.at("mode").string();
    require(mode=="prefix"||mode=="video","Unsupported evaluation mode");s.complete=mode=="video";
    const auto positive=[&](const char* key,int64_t limit) {
        const auto n=j.at(key).integer();require(n>0&&n<=limit,std::string("Invalid evaluation bound: ")+key);return n;
    };
    s.max_steps=int(positive("max_steps",64));
    s.device_bytes=positive("device_budget_bytes",1LL<<40);
    s.host_bytes=positive("host_budget_bytes",4LL<<40);
    s.output_bytes=positive("max_output_bytes",64LL<<30);
    s.latent_elements=positive("max_latent_elements",64LL<<20);
    s.video_elements=positive("max_video_elements",1LL<<30);
    s.timeout_seconds=int(positive("timeout_seconds",7200));s.fps=int(positive("fps",120));
    require(s.device_bytes>3ULL<<30,"Evaluation device budget must exceed reserved workspace");
    if(cache_declared) {
        s.weight_cache_bytes=positive("weight_cache_budget_bytes",1LL<<40);
        require(s.weight_cache_bytes<=s.device_bytes-(3ULL<<30),"Weight cache budget exceeds evaluation planning envelope");
    }
    require(s.output_bytes>=1ULL<<20,"Evaluation output budget too small for evidence");
    const auto& range=j.at("media_range").array();require(range.size()==2,"Invalid evaluation media range");
    s.video_min=float(range[0].number());s.video_max=float(range[1].number());
    require(std::isfinite(s.video_min)&&std::isfinite(s.video_max)&&s.video_min<s.video_max,"Invalid evaluation media range");
    return s;
}
void EvaluationScope::admit(uint64_t latent,uint64_t video,int steps) const {
    require(latent>0&&latent<=latent_elements,"Evaluation latent limit exceeded");
    require(video>0&&video<=video_elements,"Evaluation decoded geometry limit exceeded");
    require(steps>0&&(complete ? steps<=max_steps : max_steps<steps),"Evaluation step scope conflicts with declared schedule");
    // Reserve room for initial/final latent, conditioning and media evidence.
    // No internal tensor/solver snapshots are enabled. Child watchdog also
    // enforces actual cumulative bytes; these are not device-peak estimates.
    const auto required=latent*8+(complete?video*4:0)+(64ULL<<20);
    require(required<=output_bytes,"Evaluation output budget below declared outputs");
    require(required<=host_bytes,"Evaluation host budget below declared outputs");
}
} // namespace vrhino::product
