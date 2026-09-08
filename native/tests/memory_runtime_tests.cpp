#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vrhino/memory.h"

namespace {

using vrhino::MemoryBudget;
using vrhino::MemoryPlanner;
using vrhino::EventFenceTracker;
using vrhino::ResidencyRecord;
using vrhino::ResidencyState;
using vrhino::ResidencyTable;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

MemoryBudget budget(size_t device) {
    return {device, 256, 4096, 128, 128};
}

void tests() {
    int native_owner_releases = 0;
    {
        auto storage = std::make_shared<vrhino::Storage>();
        float value = 1.0f;
        storage->data = &value;
        storage->bytes = sizeof(value);
        storage->device = vrhino::DeviceId::accelerator(2);
        storage->domain = vrhino::MemoryDomain::Unified;
        storage->owner = true;
        storage->native_owner = std::shared_ptr<void>(
            reinterpret_cast<void*>(uintptr_t{1}),
            [&](void*) { ++native_owner_releases; });
        vrhino::Tensor tensor(storage, 0, {1}, vrhino::DType::F32);
        expect(tensor.device() == vrhino::DeviceId::accelerator(2),
               "generic device identity mismatch");
        expect(tensor.memory_domain() == vrhino::MemoryDomain::Unified,
               "tensor memory domain mismatch");
    }
    expect(native_owner_releases == 1, "opaque native owner lifetime mismatch");

    MemoryPlanner planner(budget(2048));
    expect(planner.budget().device_weight_budget_bytes() == 1792, "weight budget mismatch");
    vrhino::BackendMemoryCapabilities unified{true, true, true, true, false, true};
    planner.set_capabilities(unified);
    expect(planner.capabilities().unified_memory &&
           !planner.capabilities().explicit_transfer_required,
           "memory capabilities were not retained");
    vrhino::MemoryAccounting accounting;
    accounting.peak_host_total_bytes = 1024;
    accounting.peak_device_bytes = 768;
    expect(accounting.peak_physical_bytes(unified) == 1024,
           "unified memory was double counted");
    expect(accounting.peak_physical_bytes({}) == 1792,
           "discrete memory accounting mismatch");
    auto direct = planner.plan_admission(512, 256, {});
    expect(direct.admit && direct.evict.empty(), "direct admission failed");

    const void* a = reinterpret_cast<void*>(uintptr_t{1});
    const void* b = reinterpret_cast<void*>(uintptr_t{2});
    const void* c = reinterpret_cast<void*>(uintptr_t{3});
    std::vector<ResidencyRecord> records = {
        {a, 512, ResidencyState::Evictable, 1, 20, false, false},
        {b, 512, ResidencyState::InUse, 2, 30, true, false},
        {c, 512, ResidencyState::DevicePackedReady, 3,
         std::numeric_limits<uint64_t>::max(), false, false},
    };
    auto eviction = planner.plan_admission(1536, 700, records);
    expect(eviction.admit && eviction.evict.size() == 1 && eviction.evict[0] == c,
           "next-use eviction choice mismatch");
    auto too_large = planner.plan_admission(0, 2048, records);
    expect(!too_large.admit && too_large.rejection.find("single tensor") != std::string::npos,
           "single tensor negative gate failed");

    records[0].transfer_in_flight = true;
    records[2].transfer_in_flight = true;
    auto unsafe = planner.plan_admission(1536, 700, records);
    expect(!unsafe.admit, "in-flight/in-use eviction was accepted");

    bool invalid = false;
    try { MemoryBudget{1024, 2048, 1024, 0, 0}.validate(); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "invalid pinned budget was accepted");

    invalid = false;
    try { MemoryBudget{1024, 0, 4096, 900, 200}.validate(); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "workspace/safety overflow was accepted");

    expect(vrhino::residency_state_name(ResidencyState::TransferPending) == "TRANSFER_PENDING",
           "residency state ABI mismatch");

    invalid = false;
    try { (void)vrhino::checked_memory_add(std::numeric_limits<size_t>::max(), 1,
                                           "metadata"); }
    catch (const std::overflow_error&) { invalid = true; }
    expect(invalid, "metadata size overflow was accepted");

    invalid = false;
    try { (void)vrhino::checked_memory_multiply(std::numeric_limits<size_t>::max(), 2,
                                                "cache accounting"); }
    catch (const std::overflow_error&) { invalid = true; }
    expect(invalid, "cache accounting overflow was accepted");

    ResidencyTable table;
    table.insert({a, 512, ResidencyState::HostPageable});
    table.transition(a, ResidencyState::HostPageable, ResidencyState::TransferPending);
    invalid = false;
    try { (void)table.erase(a, ResidencyState::TransferPending); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "in-flight residency erase was accepted");
    table.transition(a, ResidencyState::TransferPending, ResidencyState::DeviceResident);
    invalid = false;
    try { table.transition(a, ResidencyState::DeviceResident, ResidencyState::HostStaging); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "illegal residency transition was accepted");
    (void)table.erase(a, ResidencyState::DeviceResident);
    invalid = false;
    try { (void)table.erase(a, ResidencyState::DeviceResident); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "double-free/stale residency erase was accepted");

    EventFenceTracker events;
    auto fence = events.create();
    expect(!events.query(fence), "unrecorded event reported complete");
    invalid = false;
    try { events.wait(fence); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "wait before record was accepted");
    events.record(fence);
    events.wait(fence);
    expect(events.query(fence), "recorded event did not report complete");
    events.destroy(fence);
    invalid = false;
    try { events.query(fence); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "query after destroy was accepted");
    invalid = false;
    try { events.destroy(fence); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "double destroy was accepted");

    EventFenceTracker other_backend;
    auto other_fence = other_backend.create();
    invalid = false;
    try { events.query(other_fence); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "cross-backend event was accepted");
    EventFenceTracker other_device(1);
    auto device_fence = other_device.create();
    auto wrong_device = device_fence;
    wrong_device.device_index = 0;
    invalid = false;
    try { other_device.query(wrong_device); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "cross-device event was accepted");
    other_backend.destroy(other_fence);
    other_device.destroy(device_fence);

    EventFenceTracker shutdown_events;
    auto shutdown_fence = shutdown_events.create();
    shutdown_events.shutdown();
    invalid = false;
    try { shutdown_events.query(shutdown_fence); }
    catch (const std::invalid_argument&) { invalid = true; }
    expect(invalid, "event use after backend shutdown was accepted");
}

}  // namespace

int main() {
    try {
        tests();
        std::cout << "backend-neutral tensor/memory/event contract cases=26 pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase10 memory tests: " << error.what() << "\n";
        return 1;
    }
}
