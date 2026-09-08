#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "vrhino/tensor.h"

namespace vrhino {

enum class ResidencyState : uint8_t {
    Unmapped,
    HostPageable,
    HostStaging,
    TransferPending,
    DeviceResident,
    DevicePackedReady,
    InUse,
    Evictable,
};

std::string residency_state_name(ResidencyState state);
size_t checked_memory_add(size_t left, size_t right, const char* context);
size_t checked_memory_multiply(size_t left, size_t right, const char* context);

struct MemoryBudget {
    size_t device_budget_bytes = 0;
    size_t host_staging_budget_bytes = 0;
    size_t host_total_budget_bytes = 0;
    size_t reserved_device_workspace_bytes = 0;
    size_t safety_margin_bytes = 0;

    void validate() const;
    size_t device_weight_budget_bytes() const;
};

struct BackendMemoryCapabilities {
    bool unified_memory = false;
    bool host_visible_device_memory = false;
    bool device_visible_host_memory = false;
    bool device_preferred = true;
    bool explicit_transfer_required = true;
    bool async_transfer_supported = true;
};

enum class EvictionPolicy : uint8_t { Lru, NextUse };

struct MemoryRuntimeOptions {
    bool enabled = false;
    bool host_staging = true;
    bool prefetch = true;
    size_t prefetch_lookahead = 1;
    EvictionPolicy eviction_policy = EvictionPolicy::NextUse;
};

struct MemoryAccounting {
    size_t vrm_mapped_bytes = 0;
    size_t host_pageable_bytes = 0;
    size_t host_staging_bytes = 0;
    size_t peak_host_staging_bytes = 0;
    size_t peak_host_total_bytes = 0;
    size_t device_resident_weight_bytes = 0;
    size_t peak_device_resident_weight_bytes = 0;
    size_t device_activation_bytes = 0;
    size_t peak_device_activation_bytes = 0;
    size_t device_workspace_bytes = 0;
    size_t quantized_packed_bytes = 0;
    size_t backend_repack_bytes = 0;
    size_t temporary_bytes = 0;
    size_t peak_temporary_bytes = 0;
    size_t peak_device_bytes = 0;

    size_t peak_physical_bytes(const BackendMemoryCapabilities& capabilities) const;
};

struct MemoryRuntimeStats {
    MemoryAccounting accounting;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t prefetch_requests = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_dropped = 0;
    uint64_t evictions = 0;
    uint64_t upload_copies = 0;
    size_t upload_bytes = 0;
    double upload_seconds = 0.0;
    double overlapped_upload_seconds = 0.0;
    uint64_t stream_waits = 0;
    uint64_t event_waits = 0;
    uint64_t forced_syncs = 0;
    uint64_t staging_hits = 0;
    uint64_t staging_misses = 0;
    uint64_t staging_evictions = 0;
    uint64_t unsafe_eviction_rejections = 0;
    uint64_t temporary_pool_reuses = 0;
    uint64_t stream_ordered_handoff_releases = 0;
    uint64_t stream_ordered_handoff_reuses = 0;
    size_t last_stream_ordered_handoff_bytes = 0;
    size_t largest_stream_ordered_handoff_bytes = 0;

    double cache_hit_rate() const;
    double prefetch_hit_rate() const;
    double upload_bandwidth_gbps() const;
    double overlap_ratio() const;
};

struct MemoryAccess {
    Tensor tensor;
    DType target_dtype = DType::F32;
    size_t canonical_bytes = 0;
    size_t backend_bytes = 0;
    bool quantized = false;
};

struct ResidencyRecord {
    const void* identity = nullptr;
    size_t bytes = 0;
    ResidencyState state = ResidencyState::HostPageable;
    uint64_t last_use = 0;
    uint64_t next_use = std::numeric_limits<uint64_t>::max();
    bool in_use = false;
    bool transfer_in_flight = false;
};

class ResidencyTable {
public:
    void insert(ResidencyRecord record);
    void transition(const void* identity, ResidencyState expected, ResidencyState target);
    ResidencyRecord erase(const void* identity, ResidencyState expected);
    bool contains(const void* identity) const;
    const ResidencyRecord& at(const void* identity) const;
    size_t size() const { return records_.size(); }
private:
    std::unordered_map<const void*, ResidencyRecord> records_;
};

struct AdmissionPlan {
    bool admit = false;
    std::vector<const void*> evict;
    std::string rejection;
};

class MemoryPlanner {
public:
    explicit MemoryPlanner(MemoryBudget budget = {}, BackendMemoryCapabilities capabilities = {});
    void set_budget(MemoryBudget budget);
    void set_capabilities(BackendMemoryCapabilities capabilities) { capabilities_ = capabilities; }
    const MemoryBudget& budget() const { return budget_; }
    const BackendMemoryCapabilities& capabilities() const { return capabilities_; }
    AdmissionPlan plan_admission(size_t resident_bytes, size_t requested_bytes,
                                 const std::vector<ResidencyRecord>& candidates,
                                 EvictionPolicy policy = EvictionPolicy::NextUse) const;
    static uint64_t next_use_after(const std::vector<MemoryAccess>& trace,
                                   size_t cursor, const void* identity);
private:
    MemoryBudget budget_;
    BackendMemoryCapabilities capabilities_;
};

// Backend-neutral handles: hardware backends may attach native state behind
// ids/opaque pointers without exposing their APIs to shared Runtime.
struct OpaqueDeviceBuffer { void* opaque = nullptr; size_t bytes = 0; };
struct TransferFence {
    uint64_t backend_id = 0;
    uint64_t id = 0;
    int32_t device_index = 0;
    bool valid() const { return backend_id != 0 && id != 0; }
};

// Shared lifecycle validator. Hardware backends pair each live handle with
// their native event object and use this tracker to reject stale/misused fences.
class EventFenceTracker {
public:
    explicit EventFenceTracker(int32_t device_index = 0);
    TransferFence create();
    void record(TransferFence fence);
    void wait(TransferFence fence) const;
    bool query(TransferFence fence) const;
    void destroy(TransferFence fence);
    void shutdown();
    uint64_t backend_id() const { return backend_id_; }
private:
    struct State { bool recorded = false; };
    const State& checked(TransferFence fence) const;
    State& checked(TransferFence fence);
    uint64_t backend_id_ = 0;
    uint64_t next_id_ = 1;
    int32_t device_index_ = 0;
    bool active_ = true;
    std::unordered_map<uint64_t, State> states_;
};

class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;
    virtual BackendMemoryCapabilities memory_capabilities() const = 0;
    virtual void configure_memory_runtime(const MemoryBudget& budget,
                                          const MemoryRuntimeOptions& options) = 0;
    virtual bool memory_runtime_enabled() const = 0;
    virtual void set_vrm_mapped_bytes(size_t bytes) = 0;
    virtual void begin_memory_trace() = 0;
    virtual std::vector<MemoryAccess> end_memory_trace() = 0;
    virtual void set_memory_trace(std::vector<MemoryAccess> trace) = 0;
    virtual MemoryRuntimeStats memory_runtime_stats() const = 0;
};

}  // namespace vrhino
