#include "vrhino/product/wan_family_conversion.h"
#include <iostream>
#include <chrono>

int main(int argc,char** argv) {
    try {
        if(argc==3 && std::string(argv[1])=="verify") {
            std::cout<<vrhino::product::canonical_json(vrhino::product::verify_wan_family_structure(argv[2]))<<'\n';return 0;
        }
        if(argc!=5) {std::cerr<<"usage: wan-family-convert MODEL_ROOT SEMANTIC_SOURCE_ROOT SPEC OUTPUT_DIR\n"
            <<"       wan-family-convert verify VRM\n";return 2;}
        uint64_t work=0;auto last=std::chrono::steady_clock::now();
        auto result=vrhino::product::convert_wan_family_package(argv[1],argv[2],argv[3],argv[4],[&](uint64_t n){
            work+=n;auto now=std::chrono::steady_clock::now();
            if(now-last>std::chrono::seconds(15)){std::cerr<<"streamed_work_bytes="<<work<<'\n';last=now;}
        });
        std::cout<<vrhino::product::canonical_json(result)<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"CONVERSION_REJECTED: "<<e.what()<<'\n';return 1;}
}
