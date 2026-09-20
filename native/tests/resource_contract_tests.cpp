#include "step_execution_test_support.h"
#include "vrhino/resource_transaction.h"
#include "vrhino/execution.h"
#include <iostream>
#include <limits>
using namespace vrhino;
namespace st=vrhino::step_test;
int main() {
    try {
        std::vector<int> order;
        try {
            ResourceRollback a([&]() noexcept { order.push_back(1); });
            ResourceRollback b([&]() noexcept { order.push_back(2); });
            throw Error("partial acquisition");
        } catch(const Error&) {}
        require(order==std::vector<int>({2,1}),"Transaction rollback order");
        { ResourceRollback committed([&]() noexcept { order.push_back(3); }); committed.commit(); }
        require(order.size()==2,"Published owner was rolled back");
        ResourceEstimate e; e.persistent_weights=1;e.resident_cache=2;e.activations=3;e.workspace=4;
        e.upload_temporary=5;e.sampling_state=6;e.component_allocations=7;e.prepared=8;
        e.source_backing=9;e.host_staging=10;
        require(e.device_peak()==36 && e.host_peak()==19,"Resource categories missing");
        ResourceAdmissionRequest{e,{36,19}}.validate();
        int rejects=0;
        auto reject=[&](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"Invalid admission accepted");++rejects;};
        reject([&]{ResourceAdmissionRequest{e,{35,19}}.validate();});
        reject([&]{ResourceAdmissionRequest{e,{36,18}}.validate();});
        reject([&]{ResourceAdmissionRequest{e,{0,19}}.validate();});
        reject([&]{auto overflow=e;overflow.activations=std::numeric_limits<size_t>::max();overflow.device_peak();});
        reject([&]{auto overflow=e;overflow.source_backing=std::numeric_limits<size_t>::max();overflow.host_peak();});
        std::weak_ptr<int> weak;
        {
            st::Backend backend;
            auto owner=std::make_shared<int>(123);weak=owner;
            backend.retain_resource_owners({owner,owner});owner.reset();
            require(!weak.expired() && backend.resource_owner_count()==1,"Explicit lease not retained/deduplicated");
            reject([&]{backend.retain_resource_owners({nullptr});});
            require(backend.resource_owner_count()==1,"Lease publication not transactional");
            reject([&]{backend.set_resource_admission({e,{35,19}});});
            require(backend.rng_calls==0 && backend.calls.empty(),"Rejected resource request touched execution");
            backend.set_resource_admission({e,{36,19}});
            st::LegacyDenoiser endpoint(backend,scalar_f32(0.1f));
            auto result=SamplingRuntime(backend).run(endpoint,st::program());
            require(result.final_latent.defined(),"Admitted request not executable");
        }
        require(weak.expired(),"Session lease leaked");
        std::cout<<"resource_contract=PASS rollback_order=reverse budget_categories=10 rejects="<<rejects<<" leases=explicit\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
