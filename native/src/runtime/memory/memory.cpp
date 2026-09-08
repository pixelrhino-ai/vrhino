#include "vrhino/memory.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace vrhino {

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

bool legal_transition(ResidencyState source, ResidencyState target) {
    switch (source) {
        case ResidencyState::Unmapped:
            return target == ResidencyState::HostPageable;
        case ResidencyState::HostPageable:
            return target == ResidencyState::HostStaging || target == ResidencyState::TransferPending ||
                   target == ResidencyState::Unmapped;
        case ResidencyState::HostStaging:
            return target == ResidencyState::HostPageable || target == ResidencyState::TransferPending;
        case ResidencyState::TransferPending:
            return target == ResidencyState::HostPageable || target == ResidencyState::HostStaging ||
                   target == ResidencyState::DeviceResident || target == ResidencyState::DevicePackedReady;
        case ResidencyState::DeviceResident:
        case ResidencyState::DevicePackedReady:
            return target == ResidencyState::InUse || target == ResidencyState::Evictable;
        case ResidencyState::InUse:
            return target == ResidencyState::DeviceResident ||
                   target == ResidencyState::DevicePackedReady || target == ResidencyState::Evictable;
        case ResidencyState::Evictable:
            return target == ResidencyState::InUse || target == ResidencyState::HostPageable ||
                   target == ResidencyState::HostStaging || target == ResidencyState::Unmapped;
    }
    return false;
}
}

std::string residency_state_name(ResidencyState state) {
    switch (state) {
        case ResidencyState::Unmapped: return "UNMAPPED";
        case ResidencyState::HostPageable: return "HOST_PAGEABLE";
        case ResidencyState::HostStaging: return "HOST_STAGING";
        case ResidencyState::TransferPending: return "TRANSFER_PENDING";
        case ResidencyState::DeviceResident: return "DEVICE_RESIDENT";
        case ResidencyState::DevicePackedReady: return "DEVICE_PACKED_READY";
        case ResidencyState::InUse: return "IN_USE";
        case ResidencyState::Evictable: return "EVICTABLE";
    }
    throw std::invalid_argument("unknown residency state");
}

size_t checked_memory_add(size_t left, size_t right, const char* context) {
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(std::string(context) + ": size addition overflow");
    return left + right;
}

size_t checked_memory_multiply(size_t left, size_t right, const char* context) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        throw std::overflow_error(std::string(context) + ": size multiplication overflow");
    return left * right;
}

void ResidencyTable::insert(ResidencyRecord record) {
    check(record.identity != nullptr, "residency identity must not be null");
    check(record.bytes > 0, "residency size must be positive");
    check(records_.emplace(record.identity, record).second,
          "duplicate/stale residency identity");
}

void ResidencyTable::transition(const void* identity, ResidencyState expected,
                                ResidencyState target) {
    auto found = records_.find(identity);
    check(found != records_.end(), "stale residency transition");
    check(found->second.state == expected, "residency transition source mismatch");
    check(legal_transition(expected, target), "illegal residency transition");
    found->second.state = target;
    found->second.transfer_in_flight = target == ResidencyState::TransferPending;
    found->second.in_use = target == ResidencyState::InUse;
}

ResidencyRecord ResidencyTable::erase(const void* identity, ResidencyState expected) {
    auto found = records_.find(identity);
    check(found != records_.end(), "double-free/stale residency erase");
    check(found->second.state == expected, "residency erase source mismatch");
    check(!found->second.in_use && !found->second.transfer_in_flight,
          "cannot erase in-use/in-flight residency");
    ResidencyRecord result = found->second;
    records_.erase(found);
    return result;
}

bool ResidencyTable::contains(const void* identity) const {
    return records_.find(identity) != records_.end();
}

const ResidencyRecord& ResidencyTable::at(const void* identity) const {
    auto found = records_.find(identity);
    check(found != records_.end(), "stale residency lookup");
    return found->second;
}

void MemoryBudget::validate() const {
    check(device_budget_bytes > 0, "device memory budget must be positive");
    check(host_total_budget_bytes > 0, "host total memory budget must be positive");
    check(host_staging_budget_bytes <= host_total_budget_bytes,
          "host staging budget exceeds host total budget");
    check(reserved_device_workspace_bytes <= device_budget_bytes,
          "reserved device workspace exceeds device budget");
    check(safety_margin_bytes <= device_budget_bytes - reserved_device_workspace_bytes,
          "device safety margin exceeds remaining device budget");
}

size_t MemoryBudget::device_weight_budget_bytes() const {
    validate();
    return device_budget_bytes - reserved_device_workspace_bytes - safety_margin_bytes;
}

size_t MemoryAccounting::peak_physical_bytes(
        const BackendMemoryCapabilities& capabilities) const {
    if (capabilities.unified_memory)
        return std::max(peak_host_total_bytes, peak_device_bytes);
    return checked_memory_add(peak_host_total_bytes, peak_device_bytes,
                              "peak physical memory");
}

double MemoryRuntimeStats::cache_hit_rate() const {
    const uint64_t total = cache_hits + cache_misses;
    return total ? static_cast<double>(cache_hits) / total : 0.0;
}
double MemoryRuntimeStats::prefetch_hit_rate() const {
    return prefetch_requests ? static_cast<double>(prefetch_hits) / prefetch_requests : 0.0;
}
double MemoryRuntimeStats::upload_bandwidth_gbps() const {
    return upload_seconds > 0.0 ? static_cast<double>(upload_bytes) / upload_seconds / 1.0e9 : 0.0;
}
double MemoryRuntimeStats::overlap_ratio() const {
    return upload_seconds > 0.0 ? std::min(1.0, overlapped_upload_seconds / upload_seconds) : 0.0;
}

MemoryPlanner::MemoryPlanner(MemoryBudget budget, BackendMemoryCapabilities capabilities)
    : budget_(budget), capabilities_(capabilities) {}

void MemoryPlanner::set_budget(MemoryBudget budget) {
    budget.validate();
    budget_ = budget;
}

AdmissionPlan MemoryPlanner::plan_admission(
    size_t resident_bytes, size_t requested_bytes,
    const std::vector<ResidencyRecord>& candidates, EvictionPolicy policy) const {
    const size_t capacity = budget_.device_weight_budget_bytes();
    if (requested_bytes > capacity)
        return {false, {}, "single tensor exceeds device weight budget"};
    if (resident_bytes > capacity)
        return {false, {}, "resident accounting exceeds device weight budget"};
    if (requested_bytes <= capacity - resident_bytes) return {true, {}, {}};

    std::vector<ResidencyRecord> safe;
    for (const ResidencyRecord& record : candidates) {
        if (!record.in_use && !record.transfer_in_flight &&
            (record.state == ResidencyState::DeviceResident ||
             record.state == ResidencyState::DevicePackedReady ||
             record.state == ResidencyState::Evictable)) safe.push_back(record);
    }
    std::sort(safe.begin(), safe.end(), [&](const ResidencyRecord& a, const ResidencyRecord& b) {
        if (policy == EvictionPolicy::Lru) return a.last_use < b.last_use;
        // Belady-like next-use choice; LRU is the stable tie-breaker. Callers
        // set next_use=MAX when the trace has no future reference.
        if (a.next_use != b.next_use) return a.next_use > b.next_use;
        return a.last_use < b.last_use;
    });
    AdmissionPlan result;
    size_t available = capacity - resident_bytes;
    for (const ResidencyRecord& record : safe) {
        result.evict.push_back(record.identity);
        available += record.bytes;
        if (requested_bytes <= available) {
            result.admit = true;
            return result;
        }
    }
    result.rejection = "insufficient evictable device residency";
    return result;
}

uint64_t MemoryPlanner::next_use_after(const std::vector<MemoryAccess>& trace,
                                       size_t cursor, const void* identity) {
    for (size_t index = cursor + 1; index < trace.size(); ++index)
        if (trace[index].tensor.data() == identity) return index;
    return std::numeric_limits<uint64_t>::max();
}

namespace {
std::atomic<uint64_t> next_backend_id{1};
}

EventFenceTracker::EventFenceTracker(int32_t device_index)
    : backend_id_(next_backend_id.fetch_add(1)), device_index_(device_index) {
    check(backend_id_ != 0, "event backend id overflow");
}

TransferFence EventFenceTracker::create() {
    check(active_, "event tracker is shut down");
    check(next_id_ != 0, "event id overflow");
    const uint64_t id = next_id_++;
    check(states_.emplace(id, State{}).second, "duplicate event id");
    return {backend_id_, id, device_index_};
}

const EventFenceTracker::State& EventFenceTracker::checked(TransferFence fence) const {
    check(active_, "event tracker is shut down");
    check(fence.valid(), "invalid event handle");
    check(fence.backend_id == backend_id_, "cross-backend event handle");
    check(fence.device_index == device_index_, "cross-device event handle");
    const auto found = states_.find(fence.id);
    check(found != states_.end(), "destroyed or stale event handle");
    return found->second;
}

EventFenceTracker::State& EventFenceTracker::checked(TransferFence fence) {
    return const_cast<State&>(static_cast<const EventFenceTracker&>(*this).checked(fence));
}

void EventFenceTracker::record(TransferFence fence) {
    State& state = checked(fence);
    check(!state.recorded, "event already recorded");
    state.recorded = true;
}

void EventFenceTracker::wait(TransferFence fence) const {
    check(checked(fence).recorded, "wait before event record");
}

bool EventFenceTracker::query(TransferFence fence) const {
    return checked(fence).recorded;
}

void EventFenceTracker::destroy(TransferFence fence) {
    (void)checked(fence);
    states_.erase(fence.id);
}

void EventFenceTracker::shutdown() {
    check(active_, "event tracker already shut down");
    states_.clear();
    active_ = false;
}

}  // namespace vrhino
