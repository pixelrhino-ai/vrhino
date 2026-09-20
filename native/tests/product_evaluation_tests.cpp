#include "vrhino/product/evaluation.h"
#include "vrhino/error.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#ifndef _WIN32
#include <unistd.h>
#include <csignal>
#endif
using namespace vrhino;namespace p=vrhino::product;namespace fs=std::filesystem;
int main(int argc,char** argv){try{
    require(argc==2,"Expected test output directory");fs::path root=argv[1];fs::create_directories(root);
    auto declaration=Json::parse(R"({"schema":"vrhino.product-evaluation.v1","precision":"bf16","precision_policy_artifact":"policy","mode":"prefix","max_steps":1,"device_budget_bytes":8589934592,"host_budget_bytes":1073741824,"max_output_bytes":134217728,"max_latent_elements":4194304,"max_video_elements":100000000,"timeout_seconds":5,"fps":16,"media_range":[-1,1]})");
    auto s=p::EvaluationScope::parse(declaration);s.admit(2096640,97044480,40);
    require(s.weight_cache_bytes==0,"Legacy cache budget changed");
    int rejects=0;const auto reject=[&](auto f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught,"Expected evaluation rejection");++rejects;};
    auto limited=declaration.object();limited["weight_cache_budget_bytes"]=Json(int64_t(1ULL<<30));
    const auto cap=p::EvaluationScope::parse(Json(limited));
    require(cap.device_bytes==s.device_bytes && cap.weight_cache_bytes==(1ULL<<30),"Cache and monitoring budgets coupled");
    for(int64_t bad:{int64_t(0),int64_t(-1),int64_t(6ULL<<30)}) {
        auto j=limited;j["weight_cache_budget_bytes"]=Json(bad);reject([&]{p::EvaluationScope::parse(Json(j));});
    }
    limited["weight_cache_budget_bytes"]=Json(1.5);reject([&]{p::EvaluationScope::parse(Json(limited));});
    for(const char* field:{"model","binding","tolerance","guidance","latent_bundle","kernel","precision_override","trace_enabled"}) {
        auto j=declaration.object();j[field]=Json(true);reject([&]{p::EvaluationScope::parse(Json(j));});
    }
    for(const char* field:{"max_steps","device_budget_bytes","host_budget_bytes","max_output_bytes","max_latent_elements","max_video_elements","timeout_seconds","fps"}) {
        auto j=declaration.object();j[field]=Json(int64_t(0));reject([&]{p::EvaluationScope::parse(Json(j));});
    }
    for(const char* field:{"schema","precision","mode","precision_policy_artifact"}) {
        auto j=declaration.object();j[field]=Json(std::string());reject([&]{p::EvaluationScope::parse(Json(j));});
    }
    reject([&]{s.admit(s.latent_elements+1,100,40);});
    reject([&]{s.admit(10,s.video_elements+1,40);});
    reject([&]{s.admit(10,100,1);});
    auto video=s;video.complete=true;video.max_steps=40;
    reject([&]{video.admit(2096640,97044480,40);}); // Output budget really enforced.
    video.output_bytes=1ULL<<30;video.admit(2096640,97044480,40);
    reject([&]{video.admit(10,100,41);});
#ifndef _WIN32
    int index=0;
    const auto test=[&](const p::EvaluationScope& scope,auto worker,auto device){
        const auto dir=root/("run-"+std::to_string(++index));fs::remove_all(dir);
        return p::supervise_evaluation(scope,dir,[&]{worker(dir);},device);
    };
    auto zero=[](){return uint64_t(0);};
    auto ok=test(s,[](auto dir){std::ofstream(dir/"done")<<"done";std::this_thread::sleep_for(std::chrono::milliseconds(150));},zero);
    require(ok.at("status").string()=="EXECUTED_UNQUALIFIED"&&!ok.at("production_numerically_qualified").boolean(),"Supervisor promoted qualification");
    int samples=0;
    auto measured=test(s,[](auto){std::this_thread::sleep_for(std::chrono::milliseconds(260));},[&]{return uint64_t(++samples*1024);});
    double integral=0,first=0,last=0,previous=0;int count=0;
    std::ifstream history(root/("run-"+std::to_string(index))/"memory-timeline.jsonl");std::string line;
    while(std::getline(history,line)) {
        const auto row=Json::parse(line);const double time=row.at("elapsed_seconds").number(),value=row.at("device_used_bytes").number();
        if(count)integral+=(time-last)*(value+previous)*0.5;else first=time;
        last=time;previous=value;++count;
    }
    require(count>=3 && measured.at("memory_sample_count").integer()==count,"Memory samples missing");
    require(std::abs(measured.at("time_weighted_mean_device_bytes").number()-integral/(last-first))<1e-6,"Time-weighted memory mean incorrect");
    auto fail=test(s,[](auto){throw Error("injected failure");},zero);require(fail.at("status").string()=="HOLD","Worker failure ignored");
    auto wait=[](auto){std::this_thread::sleep_for(std::chrono::seconds(10));};
    auto timed=s;timed.timeout_seconds=1;
    require(test(timed,wait,zero).at("stop_reason").string()=="TIME_LIMIT","Timeout not enforced");
    require(test(s,wait,[&]{return s.device_bytes+1;}).at("stop_reason").string()=="DEVICE_MEMORY_LIMIT","Device limit not enforced");
    auto small=s;small.host_bytes=1;
    require(test(small,wait,zero).at("stop_reason").string()=="HOST_MEMORY_LIMIT","RSS limit not enforced");
    require(test(s,wait,[]()->uint64_t{throw Error("sensor failure");}).at("stop_reason").string().starts_with("MONITOR_FAILED"),"Sensor failure ignored");
    auto output=s;output.output_bytes=1ULL<<20;
    require(test(output,[](auto dir){std::string data(700000,'x');std::ofstream(dir/"a")<<data;std::ofstream(dir/"b")<<data;std::this_thread::sleep_for(std::chrono::seconds(5));},zero).at("stop_reason").string()=="OUTPUT_LIMIT","Aggregate output limit ignored");
    require(test(s,[](auto){std::this_thread::sleep_for(std::chrono::milliseconds(200));kill(getppid(),SIGTERM);std::this_thread::sleep_for(std::chrono::seconds(5));},zero).at("stop_reason").string()=="CANCELLED","Cancellation ignored");
    const auto leftovers=test(s,[](auto dir){if(fork()==0){for(int i=0;i<100;++i){std::ofstream f(dir/"heartbeat",std::ios::app);f<<"x";f.close();std::this_thread::sleep_for(std::chrono::milliseconds(50));}_exit(0);}std::this_thread::sleep_for(std::chrono::milliseconds(200));},zero);
    require(leftovers.at("status").string()=="EXECUTED_UNQUALIFIED","Descendant fixture failed");
    const auto heartbeat=root/("run-"+std::to_string(index))/"heartbeat";auto size=fs::file_size(heartbeat);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));require(fs::file_size(heartbeat)==size,"Descendant survived supervisor");
    const auto existing=root/"existing";fs::create_directories(existing);std::ofstream(existing/"marker")<<"keep";
    reject([&]{p::supervise_evaluation(s,existing,[]{},zero);});require(fs::file_size(existing/"marker")==4,"Existing evidence overwritten");
#endif
    std::cout<<"PASS scope admission, "<<rejects<<" negative cases, supervisor lifecycle and limits\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
