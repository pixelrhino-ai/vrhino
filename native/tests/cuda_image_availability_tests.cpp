#include <cuda_runtime_api.h>

#include <iostream>

#include "vrhino/backend/cuda_backend.h"

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        std::cerr << "No CUDA device available for image probe\n";
        return 1;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= device_count)
        return 2;
    if (!vrhino::cuda_backend_kernel_image_available()) {
        std::cerr << "No VRhino CUDA Backend image for device " << device << '\n';
        return 3;
    }
    std::cout << "CUDA Backend image available on device " << device << '\n';
    return 0;
}
