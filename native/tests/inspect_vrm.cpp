#include <iostream>

#include "vrhino/loader.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: vrhino-vrm-inspect MODEL.vrm\n";
        return 2;
    }
    try {
        const vrhino::VrmModel model(argv[1]);
        std::cout << "architecture=" << model.architecture_id()
                  << " tensors=" << model.tensors().size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-vrm-inspect: " << error.what() << "\n";
        return 1;
    }
}
