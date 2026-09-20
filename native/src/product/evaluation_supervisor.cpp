#include "vrhino/product/evaluation.h"
#include "vrhino/error.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cerrno>
#include <csignal>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

namespace vrhino::product {
#ifndef _WIN32
namespace {
volatile sig_atomic_t interrupted=0;
void interrupt(int){interrupted=1;}
uint64_t group_rss(pid_t group) {
    uint64_t total=0;
    for(const auto& entry:std::filesystem::directory_iterator("/proc")) {
        const auto name=entry.path().filename().string();
        if(name.empty()||!std::all_of(name.begin(),name.end(),[](char c){return c>='0'&&c<='9';}))continue;
        std::ifstream f(entry.path()/"stat");std::string line;std::getline(f,line);
        const auto end=line.rfind(')');if(end==std::string::npos)continue;
        std::istringstream fields(line.substr(end+2));std::string state;long parent=0,pgrp=0;
        fields>>state>>parent>>pgrp;if(pgrp!=group)continue;
        // /proc stat field24 (RSS pages), following state/ppid/pgrp (3/4/5).
        std::string value;for(int n=6;n<=23;++n)fields>>value;
        int64_t pages=0;fields>>pages;
        require(bool(fields)&&pages>=0,"Cannot read evaluation process RSS");
        total+=uint64_t(pages)*uint64_t(sysconf(_SC_PAGESIZE));
    }
    return total;
}
uint64_t output_size(const std::filesystem::path& output) {
    uint64_t total=0,count=0;
    for(const auto& e:std::filesystem::recursive_directory_iterator(output)) {
        require(++count<=4096,"Evaluation evidence file count exceeded");
        std::error_code error;const auto status=e.symlink_status(error);
        if(error==std::errc::no_such_file_or_directory)continue; // atomic evidence replacement
        require(!error,"Cannot inspect evaluation evidence");
        require(!std::filesystem::is_symlink(status),"Symlink in evaluation evidence");
        if(std::filesystem::is_regular_file(status)) {
            const auto bytes=e.file_size(error);
            if(error==std::errc::no_such_file_or_directory)continue;
            require(!error,"Cannot measure evaluation evidence");total+=bytes;
        }
    }
    return total;
}
}
#endif
Json supervise_evaluation(const EvaluationScope& s,const std::filesystem::path& output,
    const std::function<void()>& worker,const std::function<uint64_t()>& device) {
#ifdef _WIN32
    (void)s;(void)output;(void)worker;(void)device;
    throw Error("Bounded Product evaluation requires the Linux process supervisor");
#else
    require(bool(worker)&&bool(device),"Missing evaluation supervisor callback");
    require(!std::filesystem::exists(output),"Evaluation output already exists");
    require(std::filesystem::create_directories(output),"Cannot create evaluation output");
    const auto start=std::chrono::steady_clock::now();
    const pid_t parent_pid=getpid();
    const pid_t child=fork();require(child>=0,"Cannot fork evaluation worker");
    if(child==0) {
        if(setsid()<0||prctl(PR_SET_PDEATHSIG,SIGKILL)!=0||getppid()!=parent_pid)_exit(125);
        rlimit files{rlim_t(s.output_bytes),rlim_t(s.output_bytes)};
        if(setrlimit(RLIMIT_FSIZE,&files)!=0)_exit(125);
        try { worker(); _exit(0); }
        catch(const std::exception& e) {
            try {std::ofstream f(output/"worker-error.json");
                f<<Json(Json::Object{{"status",Json(std::string("HOLD"))},{"error",Json(std::string(e.what()))}}).serialize()<<'\n';}catch(...){}
            _exit(1);
        } catch(...) {_exit(126);}
    }
    interrupted=0;
    struct sigaction action{},old_int{},old_term{};action.sa_handler=interrupt;sigemptyset(&action.sa_mask);
    sigaction(SIGINT,&action,&old_int);sigaction(SIGTERM,&action,&old_term);
    struct Cleanup {
        pid_t child; struct sigaction a,b;
        ~Cleanup(){if(child>0){kill(-child,SIGKILL);kill(child,SIGKILL);int status=0;while(waitpid(child,&status,0)<0&&errno==EINTR){}}
            sigaction(SIGINT,&a,nullptr);sigaction(SIGTERM,&b,nullptr);}
    } cleanup{child,old_int,old_term};
    uint64_t peak_host=0,peak_device=0,peak_output=0;int status=0;std::string stop;
    std::ofstream timeline(output/"memory-timeline.jsonl");
    timeline.exceptions(std::ios::badbit|std::ios::failbit);
    double first_sample=0,last_sample=0,device_integral=0,host_integral=0;
    uint64_t previous_device=0,previous_host=0,sample_count=0;
    bool exited=false;
    while(!exited) {
        try {
            const auto host_now=group_rss(child),device_now=device();
            const double now=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(sample_count) {
                const double delta=now-last_sample;
                device_integral+=delta*(double(previous_device)+double(device_now))*0.5;
                host_integral+=delta*(double(previous_host)+double(host_now))*0.5;
            }else first_sample=now;
            timeline<<Json(Json::Object{{"elapsed_seconds",Json(now)},
                {"device_used_bytes",Json(int64_t(device_now))},
                {"host_rss_bytes",Json(int64_t(host_now))}}).serialize()<<'\n';
            timeline.flush();
            ++sample_count;last_sample=now;previous_device=device_now;previous_host=host_now;
            peak_host=std::max(peak_host,host_now);
            peak_device=std::max(peak_device,device_now);
            peak_output=std::max(peak_output,output_size(output));
            if(interrupted)stop="CANCELLED";
            else if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>s.timeout_seconds)stop="TIME_LIMIT";
            else if(peak_host>s.host_bytes)stop="HOST_MEMORY_LIMIT";
            else if(peak_device>s.device_bytes)stop="DEVICE_MEMORY_LIMIT";
            else if(peak_output>s.output_bytes)stop="OUTPUT_LIMIT";
        }catch(const std::exception& e){stop=std::string("MONITOR_FAILED: ")+e.what();}
        if(!stop.empty())break;
        const pid_t waited=waitpid(child,&status,WNOHANG);
        if(waited==child){exited=true;break;}
        require(waited==0 || (waited<0&&errno==EINTR),"Evaluation wait failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if(!exited) {
        kill(-child,SIGTERM);kill(child,SIGTERM);
        // A separate process can interrupt stalled CUDA/third-party calls.
        for(int i=0;i<20;++i){if(waitpid(child,&status,WNOHANG)==child){exited=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(100));}
        if(!exited){kill(-child,SIGKILL);kill(child,SIGKILL);while(waitpid(child,&status,0)<0&&errno==EINTR){};}
    }
    // Also remove descendants left behind by a completed/failed worker.
    kill(-child,SIGKILL);
    try {peak_output=std::max(peak_output,output_size(output));if(peak_output>s.output_bytes&&stop.empty())stop="OUTPUT_LIMIT";}
    catch(const std::exception& e){if(stop.empty())stop=std::string("MONITOR_FAILED: ")+e.what();}
    const bool ok=stop.empty()&&WIFEXITED(status)&&WEXITSTATUS(status)==0;
    cleanup.child=-1;
    return Json(Json::Object{{"status",Json(std::string(ok?"EXECUTED_UNQUALIFIED":"HOLD"))},
        {"stop_reason",Json(stop)},{"child_exit_code",Json(int64_t(WIFEXITED(status)?WEXITSTATUS(status):-1))},
        {"child_signal",Json(int64_t(WIFSIGNALED(status)?WTERMSIG(status):0))},
        {"wall_seconds",Json(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count())},
        {"sampled_peak_host_rss_bytes",Json(int64_t(peak_host))},
        {"sampled_peak_device_bytes",Json(int64_t(peak_device))},
        {"memory_sample_count",Json(int64_t(sample_count))},
        {"memory_observed_seconds",Json(last_sample-first_sample)},
        {"device_byte_seconds",Json(device_integral)},
        {"host_rss_byte_seconds",Json(host_integral)},
        {"time_weighted_mean_device_bytes",Json(last_sample>first_sample?device_integral/(last_sample-first_sample):double(previous_device))},
        {"time_weighted_mean_host_rss_bytes",Json(last_sample>first_sample?host_integral/(last_sample-first_sample):double(previous_host))},
        {"memory_mean_method",Json(std::string("trapezoidal integration over recorded first-to-last samples; includes preparation; not an allocator guarantee"))},
        {"observed_output_bytes",Json(int64_t(peak_output))},
        {"sample_interval_ms",Json(int64_t(100))},
        {"host_accounting",Json(std::string("process-group RSS sum; shared pages may be counted repeatedly"))},
        {"hard_memory_bound_guaranteed",Json(false)},
        {"production_numerically_qualified",Json(false)},
        {"reference_numerically_qualified",Json(false)}});
#endif
}
} // namespace vrhino::product
