#pragma once

#include <array>
#include <set>
#include <vector>

// Test-executable-only IAT observation. Every call forwards to the original
// CUDA runtime entry point. No backend binary or NVIDIA DLL is modified.
namespace cuda_observer {
enum Kind { allocation, stream, event, pool, host, kind_count };
inline std::array<std::set<uintptr_t>, kind_count> live;
inline std::array<size_t, kind_count> created{};
inline size_t errors = 0;
inline void acquired(Kind kind, const void* pointer, cudaError_t status) {
    if (status != cudaSuccess) { ++errors; return; }
    if (!live[kind].insert(reinterpret_cast<uintptr_t>(pointer)).second) ++errors;
    ++created[kind];
}
inline void released(Kind kind, const void* pointer, cudaError_t status) {
    if (status != cudaSuccess) { ++errors; return; }
    if (pointer && live[kind].erase(reinterpret_cast<uintptr_t>(pointer)) != 1) ++errors;
}
#define VRHINO_OBSERVE_CREATE(name, signature, arguments, kind, pointer) \
    inline decltype(&name) original_##name = nullptr; \
    inline cudaError_t CUDARTAPI observe_##name signature { \
        auto status = original_##name arguments; \
        acquired(kind, status == cudaSuccess ? *pointer : nullptr, status); return status; }
#define VRHINO_OBSERVE_DESTROY(name, signature, arguments, kind, pointer) \
    inline decltype(&name) original_##name = nullptr; \
    inline cudaError_t CUDARTAPI observe_##name signature { \
        auto status = original_##name arguments; released(kind, pointer, status); return status; }
VRHINO_OBSERVE_CREATE(cudaMalloc, (void** p, size_t n), (p, n), allocation, p)
VRHINO_OBSERVE_CREATE(cudaMallocFromPoolAsync, (void** p, size_t n, cudaMemPool_t m, cudaStream_t s), (p, n, m, s), allocation, p)
VRHINO_OBSERVE_DESTROY(cudaFree, (void* p), (p), allocation, p)
VRHINO_OBSERVE_DESTROY(cudaFreeAsync, (void* p, cudaStream_t s), (p, s), allocation, p)
VRHINO_OBSERVE_CREATE(cudaHostAlloc, (void** p, size_t n, unsigned int f), (p, n, f), host, p)
VRHINO_OBSERVE_DESTROY(cudaFreeHost, (void* p), (p), host, p)
VRHINO_OBSERVE_CREATE(cudaStreamCreateWithFlags, (cudaStream_t* p, unsigned int f), (p, f), stream, p)
VRHINO_OBSERVE_DESTROY(cudaStreamDestroy, (cudaStream_t p), (p), stream, p)
VRHINO_OBSERVE_CREATE(cudaEventCreate, (cudaEvent_t* p), (p), event, p)
VRHINO_OBSERVE_CREATE(cudaEventCreateWithFlags, (cudaEvent_t* p, unsigned int f), (p, f), event, p)
VRHINO_OBSERVE_DESTROY(cudaEventDestroy, (cudaEvent_t p), (p), event, p)
VRHINO_OBSERVE_CREATE(cudaMemPoolCreate, (cudaMemPool_t* p, const cudaMemPoolProps* properties), (p, properties), pool, p)
VRHINO_OBSERVE_DESTROY(cudaMemPoolDestroy, (cudaMemPool_t p), (p), pool, p)
#undef VRHINO_OBSERVE_CREATE
#undef VRHINO_OBSERVE_DESTROY

class Imports {
    struct Patch { uintptr_t* slot; uintptr_t original; };
    std::vector<Patch> patches;
    static void replace(uintptr_t* slot, uintptr_t value) {
        DWORD previous = 0, ignored = 0;
        if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous))
            throw std::runtime_error("CUDA observer IAT protection failed");
        *slot = value;
        if (!VirtualProtect(slot, sizeof(*slot), previous, &ignored))
            throw std::runtime_error("CUDA observer IAT protection restore failed");
    }
    template<class Function> void install(const char* name, Function replacement, Function& original) {
        auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        auto* imports = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base +
            nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress);
        for (; imports->Name; ++imports) {
            if (_stricmp(reinterpret_cast<char*>(base + imports->Name), "cudart64_12.dll") != 0) continue;
            auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->OriginalFirstThunk);
            auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->FirstThunk);
            for (; names->u1.AddressOfData; ++names, ++addresses) {
                if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
                auto* entry = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
                if (std::strcmp(reinterpret_cast<char*>(entry->Name), name) != 0) continue;
                auto* slot = reinterpret_cast<uintptr_t*>(&addresses->u1.Function);
                original = reinterpret_cast<Function>(*slot);
                patches.push_back({slot, *slot});
                replace(slot, reinterpret_cast<uintptr_t>(replacement));
                return;
            }
        }
        throw std::runtime_error(std::string("CUDA observer import missing: ") + name);
    }
public:
    Imports() = default;
    Imports(const Imports&) = delete;
    Imports& operator=(const Imports&) = delete;
    ~Imports() {
        try {
            for (const auto& patch : patches) replace(patch.slot, patch.original);
        } catch (...) { std::terminate(); }
    }
    void start() {
#define VRHINO_OBSERVE(name) install(#name, &observe_##name, original_##name)
        VRHINO_OBSERVE(cudaMalloc); VRHINO_OBSERVE(cudaMallocFromPoolAsync);
        VRHINO_OBSERVE(cudaFree); VRHINO_OBSERVE(cudaFreeAsync);
        VRHINO_OBSERVE(cudaHostAlloc); VRHINO_OBSERVE(cudaFreeHost);
        VRHINO_OBSERVE(cudaStreamCreateWithFlags); VRHINO_OBSERVE(cudaStreamDestroy);
        VRHINO_OBSERVE(cudaEventCreate); VRHINO_OBSERVE(cudaEventCreateWithFlags);
        VRHINO_OBSERVE(cudaEventDestroy); VRHINO_OBSERVE(cudaMemPoolCreate);
        VRHINO_OBSERVE(cudaMemPoolDestroy);
#undef VRHINO_OBSERVE
    }
};
}
