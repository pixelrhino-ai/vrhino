#pragma once

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/product/run_session.h"

namespace vrhino::application {

inline constexpr size_t kProductJobQueueCapacity = 4;
inline constexpr size_t kProductJobRegistryCapacity = 64;

enum class JobAdmissionErrorCode {
    Conflict,
    Overloaded,
    ShuttingDown,
};

class JobAdmissionError : public std::runtime_error {
public:
    JobAdmissionError(JobAdmissionErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    JobAdmissionErrorCode code() const noexcept { return code_; }

private:
    JobAdmissionErrorCode code_;
};

struct JobFailure {
    std::string code;
    std::string message;
};

struct JobSnapshot {
    std::string id;
    std::string model_reference;
    product::RunLifecycleState state = product::RunLifecycleState::Queued;
    bool cancel_requested = false;
    std::optional<product::RunStage> stage;
    std::optional<uint64_t> completed;
    std::optional<uint64_t> total;
    std::optional<product::RunProgressUnit> unit;
    std::filesystem::path output_path;
    bool output_managed = false;
    std::optional<JobFailure> failure;
};

enum class JobCancelDisposition {
    QueuedCancelled,
    RunningCancellationRequested,
    TerminalUnchanged,
};

struct JobCancelResult {
    JobCancelDisposition disposition = JobCancelDisposition::TerminalUnchanged;
    JobSnapshot snapshot;
};

// Copied, transport-neutral access to one job's bounded RunSession history.
// A queued job has session_started=false and no events. No references into the
// registry or RunSession are exposed to callers.
struct JobEventSnapshot {
    std::vector<product::RunEvent> events;
    product::RunLifecycleState state = product::RunLifecycleState::Queued;
    uint64_t first_available_sequence = 0;
    uint64_t latest_sequence = 0;
    bool history_truncated = false;
    bool session_started = false;
    bool manager_stopping = false;
};

// Process-memory-only, single-worker Product admission and execution. HTTP is
// a caller, not a dependency; the injected executor is the same seam consumed
// by RunSession and by deterministic CPU tests.
class JobManager {
public:
    JobManager(std::filesystem::path output_root,
               product::RunExecutor executor,
               uint64_t process_nonce = 0);
    ~JobManager();
    JobManager(const JobManager&) = delete;
    JobManager& operator=(const JobManager&) = delete;

    JobSnapshot submit(product::ResolvedRunnableModel model,
                       product::RunOptions options);
    std::optional<JobSnapshot> snapshot(const std::string& id) const;
    std::optional<JobEventSnapshot> events_after(
        const std::string& id, uint64_t sequence) const;
    std::optional<JobEventSnapshot> wait_for_events(
        const std::string& id, uint64_t sequence,
        std::chrono::milliseconds maximum_wait);
    std::optional<JobCancelResult> cancel(const std::string& id);
    void shutdown() noexcept;

    const std::filesystem::path& output_root() const noexcept;
    uint64_t process_nonce() const noexcept;

    static bool valid_job_id(const std::string& id) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vrhino::application
