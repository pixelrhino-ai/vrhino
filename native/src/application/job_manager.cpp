#include "vrhino/application/job_manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

#include <unistd.h>

namespace vrhino::application {
namespace fs = std::filesystem;
namespace {

bool terminal(const product::RunLifecycleState state) noexcept {
    return state == product::RunLifecycleState::Succeeded ||
           state == product::RunLifecycleState::Failed ||
           state == product::RunLifecycleState::Cancelled;
}

uint64_t generate_process_nonce() {
    std::random_device random;
    const uint64_t random_bits =
        (static_cast<uint64_t>(random()) << 32) ^ random();
    const uint64_t clock_bits = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return random_bits ^ clock_bits ^
           (static_cast<uint64_t>(::getpid()) << 17);
}

std::string format_job_id(const uint64_t nonce, const uint64_t counter) {
    std::ostringstream result;
    result << 'r' << std::hex << std::setfill('0')
           << std::setw(16) << nonce << '-'
           << std::setw(16) << counter;
    return result.str();
}

bool path_exists_no_follow(const fs::path& path) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory) return false;
    if (error)
        throw JobAdmissionError(JobAdmissionErrorCode::Conflict,
                                "output destination cannot be inspected");
    return status.type() != fs::file_type::not_found;
}

fs::path normalized_absolute_output(const fs::path& requested) {
    if (requested.empty() || !requested.is_absolute())
        throw JobAdmissionError(JobAdmissionErrorCode::Conflict,
                                "output destination must be an absolute path");
    std::error_code error;
    fs::path result = fs::weakly_canonical(requested, error);
    if (error) {
        error.clear();
        result = fs::absolute(requested, error).lexically_normal();
    }
    if (error || result.empty() || !result.is_absolute())
        throw JobAdmissionError(JobAdmissionErrorCode::Conflict,
                                "output destination cannot be normalized");
    return result;
}

JobFailure safe_failure(const product::RunSessionSnapshot& session) {
    if (!session.failure)
        return JobFailure{"execution_failed", "Product execution failed"};
    if (!session.failure->code)
        return JobFailure{"execution_failed", "Product execution failed"};
    switch (*session.failure->code) {
        case product::ModelPackageErrorCode::InvalidInput:
            return {"invalid_input", "Product input validation failed"};
        case product::ModelPackageErrorCode::UnsupportedGpu:
        case product::ModelPackageErrorCode::InsufficientVram:
        case product::ModelPackageErrorCode::DriverIncompatible:
            return {"backend_unavailable",
                    "Product execution backend is unavailable"};
        case product::ModelPackageErrorCode::OutOfMemory:
            return {"out_of_memory", "Product execution ran out of memory"};
        case product::ModelPackageErrorCode::OutputExists:
        case product::ModelPackageErrorCode::OutputInvalid:
        case product::ModelPackageErrorCode::VideoEncodingFailed:
            return {"output_failed", "Product output could not be published"};
        case product::ModelPackageErrorCode::Cancelled:
            return {"cancelled", "Product execution was cancelled"};
        default:
            return {"execution_failed", "Product execution failed"};
    }
}

}  // namespace

class JobManager::Impl {
public:
    struct JobRecord {
        std::string id;
        product::ResolvedRunnableModel model;
        product::RunOptions options;
        fs::path output;
        bool output_managed = false;
        product::RunLifecycleState queued_state =
            product::RunLifecycleState::Queued;
        bool queued_cancel_requested = false;
        bool terminal_recorded = false;
        std::shared_ptr<product::RunSession> session;
    };

    Impl(fs::path root, product::RunExecutor run_executor,
         const uint64_t requested_nonce)
        : output_root(std::move(root)),
          executor(std::move(run_executor)),
          nonce(requested_nonce == 0 ? generate_process_nonce()
                                     : requested_nonce) {
        if (!executor)
            throw std::invalid_argument("JobManager executor is required");
        std::error_code error;
        output_root = fs::absolute(output_root, error).lexically_normal();
        if (error || output_root.empty())
            throw std::invalid_argument("JobManager output root is invalid");
        worker = std::thread([this] { worker_loop(); });
    }

    ~Impl() { shutdown(); }

    JobSnapshot snapshot_locked(const std::shared_ptr<JobRecord>& job) const {
        JobSnapshot result;
        result.id = job->id;
        result.model_reference = job->model.manifest.identity.reference();
        result.output_path = job->output;
        result.output_managed = job->output_managed;
        if (!job->session) {
            result.state = job->queued_state;
            result.cancel_requested = job->queued_cancel_requested;
            return result;
        }
        const product::RunSessionSnapshot session = job->session->snapshot();
        result.state = session.state;
        result.cancel_requested = session.cancel_requested;
        result.stage = session.stage;
        result.completed = session.completed;
        result.total = session.total;
        result.unit = session.unit;
        if (session.state == product::RunLifecycleState::Failed)
            result.failure = safe_failure(session);
        return result;
    }

    JobEventSnapshot event_snapshot_locked(
        const std::shared_ptr<JobRecord>& job,
        const uint64_t sequence) const {
        JobEventSnapshot result;
        result.manager_stopping = stopping;
        if (!job->session) {
            result.state = job->queued_state;
            return result;
        }

        result.session_started = true;
        // Read lifecycle first. If a new event arrives between these copied
        // snapshots, retaining an older non-terminal state causes one extra
        // wait iteration; it can never close a stream before that event.
        result.state = job->session->snapshot().state;
        const product::RunEventHistorySnapshot history =
            job->session->events_after(sequence);
        result.events = history.events;
        result.first_available_sequence = history.first_retained_sequence;
        result.latest_sequence = history.latest_sequence;
        result.history_truncated = history.history_truncated;
        return result;
    }

    void notify_event_waiters() noexcept {
        event_generation.fetch_add(1, std::memory_order_release);
    }

    void release_output_locked(const std::shared_ptr<JobRecord>& job) {
        const auto found = output_reservations.find(job->output.string());
        if (found != output_reservations.end() && found->second == job->id)
            output_reservations.erase(found);
    }

    void record_terminal_locked(const std::shared_ptr<JobRecord>& job) {
        if (job->terminal_recorded) return;
        job->terminal_recorded = true;
        release_output_locked(job);
        terminal_order.push_back(job->id);
    }

    void prune_terminal_locked() {
        while (jobs.size() >= kProductJobRegistryCapacity &&
               !terminal_order.empty()) {
            const std::string id = terminal_order.front();
            terminal_order.pop_front();
            const auto found = jobs.find(id);
            if (found != jobs.end()) jobs.erase(found);
        }
        if (jobs.size() >= kProductJobRegistryCapacity)
            throw JobAdmissionError(JobAdmissionErrorCode::Overloaded,
                                    "job registry is at capacity");
    }

    void worker_loop() noexcept {
        for (;;) {
            std::shared_ptr<JobRecord> job;
            std::shared_ptr<product::RunSession> session;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [this] {
                    return stopping || !queue.empty();
                });
                if (stopping && queue.empty()) return;
                const std::string id = queue.front();
                queue.pop_front();
                const auto found = jobs.find(id);
                if (found == jobs.end() ||
                    found->second->queued_state ==
                        product::RunLifecycleState::Cancelled)
                    continue;
                job = found->second;
                session = std::make_shared<product::RunSession>(
                    job->model, job->options, executor,
                    [this](const product::RunEvent&) noexcept {
                        // Notification only: RunSession already retained the
                        // value. Product execution never waits for a client or
                        // performs transport work here.
                        notify_event_waiters();
                    });
                job->session = session;
                active_id = id;
            }

            try {
                (void)session->run();
            } catch (...) {
                // RunSession owns the complete terminal classification.
            }

            {
                std::lock_guard lock(mutex);
                record_terminal_locked(job);
                if (active_id == job->id) active_id.clear();
            }
            condition.notify_all();
            notify_event_waiters();
        }
    }

    void shutdown() noexcept {
        std::shared_ptr<product::RunSession> active;
        {
            std::lock_guard lock(mutex);
            if (stopping) {
                if (!worker.joinable()) return;
            } else {
                stopping = true;
                for (const std::string& id : queue) {
                    const auto found = jobs.find(id);
                    if (found == jobs.end()) continue;
                    found->second->queued_state =
                        product::RunLifecycleState::Cancelled;
                    found->second->queued_cancel_requested = true;
                    record_terminal_locked(found->second);
                }
                queue.clear();
                if (!active_id.empty()) {
                    const auto found = jobs.find(active_id);
                    if (found != jobs.end()) active = found->second->session;
                }
            }
        }
        if (active) active->request_cancel();
        condition.notify_all();
        notify_event_waiters();
        if (worker.joinable()) worker.join();
    }

    fs::path output_root;
    product::RunExecutor executor;
    uint64_t nonce;
    std::atomic<uint64_t> next_counter{1};

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::atomic<uint64_t> event_generation{0};
    std::map<std::string, std::shared_ptr<JobRecord>> jobs;
    std::deque<std::string> queue;
    std::deque<std::string> terminal_order;
    std::map<std::string, std::string> output_reservations;
    std::string active_id;
    bool stopping = false;
    std::thread worker;
};

JobManager::JobManager(fs::path output_root, product::RunExecutor executor,
                       const uint64_t process_nonce)
    : impl_(std::make_unique<Impl>(std::move(output_root),
                                   std::move(executor), process_nonce)) {}

JobManager::~JobManager() = default;

JobSnapshot JobManager::submit(product::ResolvedRunnableModel model,
                               product::RunOptions options) {
    const bool managed = options.output.empty();
    fs::path explicit_output;
    if (!managed) explicit_output = normalized_absolute_output(options.output);

    std::lock_guard lock(impl_->mutex);
    if (impl_->stopping)
        throw JobAdmissionError(JobAdmissionErrorCode::ShuttingDown,
                                "job manager is shutting down");
    if (impl_->queue.size() >= kProductJobQueueCapacity)
        throw JobAdmissionError(JobAdmissionErrorCode::Overloaded,
                                "Product job queue is full");
    impl_->prune_terminal_locked();

    if (!managed) {
        fs::path partial = explicit_output;
        partial += ".partial";
        if (path_exists_no_follow(explicit_output) ||
            path_exists_no_follow(partial) ||
            impl_->output_reservations.contains(explicit_output.string()))
            throw JobAdmissionError(JobAdmissionErrorCode::Conflict,
                                    "output destination is already in use");
    }

    const uint64_t counter =
        impl_->next_counter.fetch_add(1, std::memory_order_relaxed);
    if (counter == 0 || counter == std::numeric_limits<uint64_t>::max())
        throw JobAdmissionError(JobAdmissionErrorCode::Overloaded,
                                "job identifier space is exhausted");
    const std::string id = format_job_id(impl_->nonce, counter);
    const fs::path output = managed
        ? (impl_->output_root / id / "output.mp4") : explicit_output;
    if (managed && (path_exists_no_follow(output) ||
                    impl_->output_reservations.contains(output.string())))
        throw JobAdmissionError(JobAdmissionErrorCode::Conflict,
                                "generated output destination is already in use");

    options.output = output;
    options.overwrite = false;
    auto job = std::make_shared<Impl::JobRecord>();
    job->id = id;
    job->model = std::move(model);
    job->options = std::move(options);
    job->output = output;
    job->output_managed = managed;
    impl_->output_reservations.emplace(output.string(), id);
    impl_->jobs.emplace(id, job);
    impl_->queue.push_back(id);
    const JobSnapshot result = impl_->snapshot_locked(job);
    impl_->condition.notify_one();
    return result;
}

std::optional<JobSnapshot> JobManager::snapshot(const std::string& id) const {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end()) return std::nullopt;
    return impl_->snapshot_locked(found->second);
}

std::optional<JobEventSnapshot> JobManager::events_after(
    const std::string& id, const uint64_t sequence) const {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end()) return std::nullopt;
    return impl_->event_snapshot_locked(found->second, sequence);
}

std::optional<JobEventSnapshot> JobManager::wait_for_events(
    const std::string& id, const uint64_t sequence,
    const std::chrono::milliseconds maximum_wait) {
    const uint64_t generation =
        impl_->event_generation.load(std::memory_order_acquire);
    std::optional<JobEventSnapshot> current = events_after(id, sequence);
    if (!current || !current->events.empty() || terminal(current->state) ||
        current->manager_stopping)
        return current;

    constexpr std::chrono::milliseconds kWaitSlice{25};
    const auto deadline = std::chrono::steady_clock::now() + maximum_wait;
    while (std::chrono::steady_clock::now() < deadline &&
           impl_->event_generation.load(std::memory_order_acquire) ==
               generation) {
        const auto remaining = std::chrono::duration_cast<
            std::chrono::milliseconds>(deadline -
                                      std::chrono::steady_clock::now());
        std::this_thread::sleep_for(std::min(kWaitSlice, remaining));
    }
    return events_after(id, sequence);
}

std::optional<JobCancelResult> JobManager::cancel(const std::string& id) {
    std::shared_ptr<product::RunSession> session;
    {
        std::lock_guard lock(impl_->mutex);
        const auto found = impl_->jobs.find(id);
        if (found == impl_->jobs.end()) return std::nullopt;
        const std::shared_ptr<Impl::JobRecord>& job = found->second;
        if (!job->session) {
            if (terminal(job->queued_state))
                return JobCancelResult{
                    JobCancelDisposition::TerminalUnchanged,
                    impl_->snapshot_locked(job)};
            const auto queued = std::find(
                impl_->queue.begin(), impl_->queue.end(), id);
            if (queued != impl_->queue.end()) impl_->queue.erase(queued);
            job->queued_state = product::RunLifecycleState::Cancelled;
            job->queued_cancel_requested = true;
            impl_->record_terminal_locked(job);
            impl_->notify_event_waiters();
            return JobCancelResult{
                JobCancelDisposition::QueuedCancelled,
                impl_->snapshot_locked(job)};
        }
        const product::RunSessionSnapshot current = job->session->snapshot();
        if (terminal(current.state))
            return JobCancelResult{
                JobCancelDisposition::TerminalUnchanged,
                impl_->snapshot_locked(job)};
        session = job->session;
    }

    session->request_cancel();
    const std::optional<JobSnapshot> current = snapshot(id);
    return JobCancelResult{
        JobCancelDisposition::RunningCancellationRequested,
        *current};
}

void JobManager::shutdown() noexcept { impl_->shutdown(); }

const fs::path& JobManager::output_root() const noexcept {
    return impl_->output_root;
}

uint64_t JobManager::process_nonce() const noexcept { return impl_->nonce; }

bool JobManager::valid_job_id(const std::string& id) noexcept {
    if (id.size() != 34 || id.front() != 'r' || id[17] != '-') return false;
    for (size_t index = 1; index < id.size(); ++index) {
        if (index == 17) continue;
        const char value = id[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f')))
            return false;
    }
    return true;
}

}  // namespace vrhino::application
