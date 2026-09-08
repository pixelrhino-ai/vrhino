#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "vrhino/product/run.h"

namespace vrhino::product {

struct RunFailure {
    std::optional<ModelPackageErrorCode> code;
    std::string message;
};

struct RunSessionSnapshot {
    RunLifecycleState state = RunLifecycleState::Queued;
    bool cancel_requested = false;
    std::optional<RunStage> stage;
    std::optional<uint64_t> completed;
    std::optional<uint64_t> total;
    std::optional<RunProgressUnit> unit;
    std::optional<RunResult> result;
    std::optional<RunFailure> failure;
    uint64_t latest_event_sequence = 0;
};

struct RunEventHistorySnapshot {
    std::vector<RunEvent> events;
    uint64_t first_retained_sequence = 0;
    uint64_t latest_sequence = 0;
    bool history_truncated = false;
};

using RunExecutor = std::function<RunResult(
    const ResolvedRunnableModel&, const RunOptions&, RunEventSink)>;

// Synchronous, single-use Product execution state. The caller owns the thread
// that invokes run(); control/status callers may concurrently request
// cancellation and read value snapshots.
class RunSession {
public:
    static constexpr size_t kEventHistoryCapacity = 256;

    RunSession(ResolvedRunnableModel model, RunOptions options,
               RunExecutor executor, RunEventSink event_sink = {});
    RunSession(const RunSession&) = delete;
    RunSession& operator=(const RunSession&) = delete;

    RunResult run();
    void request_cancel();

    bool cancel_requested() const noexcept;
    RunSessionSnapshot snapshot() const;
    RunEventHistorySnapshot events_after(uint64_t sequence) const;
    std::filesystem::path effective_output_path() const;

private:
    static bool terminal(RunLifecycleState state) noexcept;
    RunEvent append_event_locked(RunEvent event);
    void deliver_event(const RunEvent& event) const noexcept;
    void emit_executor_event(const RunEvent& event);
    void finish(RunLifecycleState state, std::optional<RunFailure> failure,
                std::optional<RunResult> result);

    ResolvedRunnableModel model_;
    RunOptions options_;
    RunExecutor executor_;
    RunEventSink event_sink_;
    std::filesystem::path effective_output_;
    std::atomic<bool> cancel_requested_{false};

    mutable std::mutex mutex_;
    RunLifecycleState state_ = RunLifecycleState::Queued;
    bool run_invoked_ = false;
    uint64_t next_sequence_ = 1;
    std::deque<RunEvent> events_;
    std::optional<RunStage> stage_;
    std::optional<uint64_t> completed_;
    std::optional<uint64_t> total_;
    std::optional<RunProgressUnit> unit_;
    std::optional<RunResult> result_;
    std::optional<RunFailure> failure_;
};

}  // namespace vrhino::product
