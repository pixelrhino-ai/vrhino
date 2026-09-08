#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vrhino/application/job_manager.h"
#include "vrhino/product/model_package.h"

namespace fs = std::filesystem;
namespace application = vrhino::application;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

product::ResolvedRunnableModel fake_model() {
    product::ResolvedRunnableModel result;
    result.manifest = product::load_model_package_manifest(
        fs::path(VRHINO_TEST_SOURCE_ROOT) /
        "specs/ltx_v0_9_1/successors/1.1.1/vrhino-model.json");
    return result;
}

product::RunOptions options(const std::string& prompt,
                            const fs::path& output = {}) {
    product::RunOptions result;
    result.model_reference = "vrhino/ltx-video-v0.9.1:1.1.1";
    result.prompt = prompt;
    result.seed = 5703;
    result.output = output;
    return result;
}

application::JobSnapshot wait_terminal(application::JobManager& manager,
                                       const std::string& id) {
    for (size_t attempt = 0; attempt < 5000; ++attempt) {
        const auto current = manager.snapshot(id);
        require_test(current.has_value(), "job disappeared while waiting");
        if (current->state == product::RunLifecycleState::Succeeded ||
            current->state == product::RunLifecycleState::Failed ||
            current->state == product::RunLifecycleState::Cancelled)
            return *current;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("job did not reach terminal state");
}

void wait_running(application::JobManager& manager, const std::string& id) {
    for (size_t attempt = 0; attempt < 5000; ++attempt) {
        const auto current = manager.snapshot(id);
        require_test(current.has_value(), "job disappeared while waiting to run");
        if (current->state == product::RunLifecycleState::Running) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("job did not start");
}

struct FakeExecutor {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> calls;
    std::set<std::string> blocked;
    std::set<std::string> released;
    size_t active = 0;
    size_t maximum_active = 0;

    product::RunExecutor callback() {
        return [this](const product::ResolvedRunnableModel& model,
                      const product::RunOptions& run,
                      product::RunEventSink events) {
            {
                std::lock_guard lock(mutex);
                calls.push_back(run.prompt);
                ++active;
                maximum_active = std::max(maximum_active, active);
            }
            condition.notify_all();
            const auto finish = [this] {
                std::lock_guard lock(mutex);
                --active;
                condition.notify_all();
            };
            if (events) {
                events(product::RunEvent::stage_changed(
                    product::RunStage::Sampling, "fake sampling"));
                events(product::RunEvent::progress(
                    product::RunStage::Sampling, 1, 2,
                    product::RunProgressUnit::Step));
                if (run.prompt == "many-events") {
                    for (uint64_t index = 0; index < 300; ++index)
                        events(product::RunEvent::progress(
                            product::RunStage::Sampling, index + 1, 300,
                            product::RunProgressUnit::Step));
                }
            }
            for (;;) {
                bool wait = false;
                {
                    std::lock_guard lock(mutex);
                    wait = blocked.contains(run.prompt) &&
                           !released.contains(run.prompt);
                }
                if (!wait) break;
                if (run.cancellation_requested && run.cancellation_requested()) {
                    finish();
                    throw product::ModelPackageError(
                        product::ModelPackageErrorCode::Cancelled,
                        "fake execution cancelled");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (run.cancellation_requested && run.cancellation_requested()) {
                finish();
                throw product::ModelPackageError(
                    product::ModelPackageErrorCode::Cancelled,
                    "fake execution cancelled");
            }
            if (run.prompt == "fail") {
                finish();
                throw std::runtime_error("private fake failure detail");
            }
            fs::create_directories(run.output.parent_path());
            std::ofstream output(run.output, std::ios::binary);
            output << "fake";
            output.close();
            product::RunResult result;
            result.identity = model.manifest.identity;
            result.seed = run.seed.value_or(0);
            result.output_bytes = 4;
            finish();
            return result;
        };
    }

    void block(const std::string& prompt) {
        std::lock_guard lock(mutex);
        blocked.insert(prompt);
    }

    void release(const std::string& prompt) {
        std::lock_guard lock(mutex);
        released.insert(prompt);
        condition.notify_all();
    }

    bool called(const std::string& prompt) {
        std::lock_guard lock(mutex);
        return std::find(calls.begin(), calls.end(), prompt) != calls.end();
    }
};

template <typename Operation>
void admission_rejects(Operation&& operation,
                       const application::JobAdmissionErrorCode code,
                       const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": admission unexpectedly succeeded");
    } catch (const application::JobAdmissionError& error) {
        require_test(error.code() == code, context + ": wrong error code");
    }
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-job-manager-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        fs::create_directories(root);
        FakeExecutor fake;
        fake.block("active");
        application::JobManager manager(
            root / "runs", fake.callback(), 0x123456789abcdef0ULL);

        const application::JobSnapshot active =
            manager.submit(fake_model(), options("active"));
        require_test(application::JobManager::valid_job_id(active.id) &&
                         active.output_managed &&
                         active.output_path ==
                             root / "runs" / active.id / "output.mp4",
                     "managed output/job id contract drift");

        for (const std::string& hostile_id : {
                 std::string(), std::string("r"),
                 std::string("r123456789abcdef-0000000000000001"),
                 std::string("r0123456789abcdef00000000000000001"),
                 std::string("r0123456789abcdef-000000000000000"),
                 std::string("r0123456789abcdef-00000000000000000"),
                 std::string("r0123456789ABCDEF-0000000000000001"),
                 std::string("r0123456789abcdef-000000000000000g"),
                 std::string("r0123456789abcdef/0000000000000001"),
                 std::string("r0123456789abcdef-%000000000000001")}) {
            require_test(!application::JobManager::valid_job_id(hostile_id),
                         "hostile Job ID was accepted");
        }
        const auto id_oracle = [](const std::string& id) {
            if (id.size() != 34 || id.front() != 'r' || id[17] != '-')
                return false;
            for (size_t index = 1; index < id.size(); ++index) {
                if (index == 17) continue;
                const char value = id[index];
                if (!((value >= '0' && value <= '9') ||
                      (value >= 'a' && value <= 'f')))
                    return false;
            }
            return true;
        };
        constexpr char id_bytes[] =
            "r-0123456789abcdefABCDEF%/._abcdefghijklmnopqrstuvwxyz";
        uint64_t id_state = 0xa54ff53a5f1d36f1ULL;
        for (size_t iteration = 0; iteration < 4096; ++iteration) {
            id_state ^= id_state << 13;
            id_state ^= id_state >> 7;
            id_state ^= id_state << 17;
            const size_t length = static_cast<size_t>(id_state % 129);
            std::string candidate;
            candidate.reserve(length);
            for (size_t index = 0; index < length; ++index) {
                id_state ^= id_state << 13;
                id_state ^= id_state >> 7;
                id_state ^= id_state << 17;
                candidate.push_back(id_bytes[
                    id_state % (sizeof(id_bytes) - 1)]);
            }
            require_test(application::JobManager::valid_job_id(candidate) ==
                             id_oracle(candidate),
                         "Job ID fuzz disagreed with the closed grammar");
        }
        wait_running(manager, active.id);
        std::optional<application::JobEventSnapshot> active_events;
        for (size_t attempt = 0; attempt < 5000; ++attempt) {
            active_events = manager.events_after(active.id, 0);
            if (active_events && active_events->latest_sequence >= 3) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require_test(active_events && active_events->session_started &&
                         active_events->latest_sequence >= 3 &&
                         !active_events->history_truncated,
                     "copied RunSession event access drift");

        std::vector<application::JobSnapshot> queued;
        for (size_t index = 0; index < 4; ++index)
            queued.push_back(manager.submit(
                fake_model(), options("queued-" + std::to_string(index))));
        admission_rejects([&] {
            (void)manager.submit(fake_model(), options("overloaded"));
        }, application::JobAdmissionErrorCode::Overloaded, "queue bound");

        const auto queued_cancel = manager.cancel(queued[1].id);
        require_test(queued_cancel &&
                         queued_cancel->disposition ==
                             application::JobCancelDisposition::QueuedCancelled &&
                         queued_cancel->snapshot.state ==
                             product::RunLifecycleState::Cancelled,
                     "queued cancellation contract drift");
        const application::JobSnapshot replacement =
            manager.submit(fake_model(), options("replacement"));

        const auto running_cancel = manager.cancel(active.id);
        require_test(running_cancel &&
                         running_cancel->disposition ==
                             application::JobCancelDisposition::
                                 RunningCancellationRequested,
                     "running cancellation did not return pending disposition");
        const auto cancellation_events = manager.wait_for_events(
            active.id, active_events->latest_sequence,
            std::chrono::seconds(1));
        require_test(cancellation_events &&
                         cancellation_events->state ==
                             product::RunLifecycleState::Cancelled &&
                         !cancellation_events->events.empty() &&
                         cancellation_events->events.back().state ==
                             product::RunLifecycleState::Cancelled,
                     "event wait did not observe terminal cancellation");
        const application::JobSnapshot cancelled =
            wait_terminal(manager, active.id);
        require_test(cancelled.state == product::RunLifecycleState::Cancelled &&
                         cancelled.cancel_requested &&
                         !fs::exists(cancelled.output_path) &&
                         !fs::exists(cancelled.output_path.string() + ".partial"),
                     "running cancellation output/state drift");

        for (const application::JobSnapshot& job : queued) {
            const application::JobSnapshot terminal =
                wait_terminal(manager, job.id);
            if (job.id == queued[1].id)
                require_test(terminal.state ==
                                 product::RunLifecycleState::Cancelled,
                             "queued cancelled job changed terminal state");
            else
                require_test(terminal.state ==
                                 product::RunLifecycleState::Succeeded,
                             "queued job did not execute successfully");
        }
        require_test(wait_terminal(manager, replacement.id).state ==
                         product::RunLifecycleState::Succeeded &&
                         !fake.called("queued-1") && fake.maximum_active == 1,
                     "queued cancellation leaked or Product execution overlapped");

        const application::JobSnapshot failed = manager.submit(
            fake_model(), options("fail"));
        const application::JobSnapshot failed_terminal =
            wait_terminal(manager, failed.id);
        require_test(failed_terminal.state == product::RunLifecycleState::Failed &&
                         failed_terminal.failure &&
                         failed_terminal.failure->code == "execution_failed" &&
                         failed_terminal.failure->message.find("private") ==
                             std::string::npos,
                     "safe fake failure mapping drift");
        const auto terminal_cancel = manager.cancel(failed.id);
        require_test(terminal_cancel &&
                         terminal_cancel->disposition ==
                             application::JobCancelDisposition::TerminalUnchanged &&
                         terminal_cancel->snapshot.state ==
                             product::RunLifecycleState::Failed,
                     "terminal DELETE mutated job state");

        const application::JobSnapshot many = manager.submit(
            fake_model(), options("many-events"));
        require_test(wait_terminal(manager, many.id).state ==
                         product::RunLifecycleState::Succeeded,
                     "many-event job did not succeed");
        const auto truncated = manager.events_after(many.id, 0);
        require_test(truncated && truncated->history_truncated &&
                         truncated->events.size() ==
                             product::RunSession::kEventHistoryCapacity &&
                         truncated->first_available_sequence > 1 &&
                         truncated->events.front().sequence ==
                             truncated->first_available_sequence &&
                         truncated->events.back().sequence ==
                             truncated->latest_sequence,
                     "bounded event history metadata drift");
        const auto retained = manager.events_after(
            many.id, truncated->first_available_sequence - 1);
        require_test(retained && !retained->history_truncated &&
                         retained->events.size() ==
                             product::RunSession::kEventHistoryCapacity,
                     "valid retained event cursor was rejected");

        fake.block("collision-owner");
        const fs::path shared_output = root / "explicit" / "output.mp4";
        const application::JobSnapshot owner = manager.submit(
            fake_model(), options("collision-owner", shared_output));
        wait_running(manager, owner.id);
        admission_rejects([&] {
            (void)manager.submit(
                fake_model(), options("collision-second", shared_output));
        }, application::JobAdmissionErrorCode::Conflict,
        "reserved output collision");
        (void)manager.cancel(owner.id);
        require_test(wait_terminal(manager, owner.id).state ==
                         product::RunLifecycleState::Cancelled,
                     "collision owner did not cancel");

        fs::create_directories(shared_output.parent_path());
        std::ofstream(shared_output) << "existing";
        admission_rejects([&] {
            (void)manager.submit(
                fake_model(), options("existing-output", shared_output));
        }, application::JobAdmissionErrorCode::Conflict,
        "existing output collision");
        require_test(std::ifstream(shared_output).good(),
                     "existing output was overwritten");

        FakeExecutor concurrent_fake;
        concurrent_fake.block("concurrent");
        application::JobManager concurrent(
            root / "concurrent-runs", concurrent_fake.callback(), 7);
        const auto concurrent_job = concurrent.submit(
            fake_model(), options("concurrent"));
        wait_running(concurrent, concurrent_job.id);
        std::optional<application::JobEventSnapshot> concurrent_baseline;
        for (size_t attempt = 0; attempt < 5000; ++attempt) {
            concurrent_baseline = concurrent.events_after(concurrent_job.id, 0);
            if (concurrent_baseline &&
                concurrent_baseline->latest_sequence >= 3)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require_test(concurrent_baseline &&
                         concurrent_baseline->latest_sequence >= 3,
                     "concurrent event baseline missing");
        std::atomic<bool> event_wait_failed{false};
        std::vector<std::thread> event_waiters;
        for (size_t waiter = 0; waiter < 4; ++waiter) {
            event_waiters.emplace_back([&] {
                const auto update = concurrent.wait_for_events(
                    concurrent_job.id,
                    concurrent_baseline->latest_sequence,
                    std::chrono::seconds(2));
                if (!update || update->events.empty())
                    event_wait_failed.store(true, std::memory_order_release);
            });
        }
        std::atomic<bool> readers_done{false};
        std::atomic<bool> reader_failed{false};
        std::vector<std::thread> readers;
        for (size_t reader = 0; reader < 4; ++reader) {
            readers.emplace_back([&] {
                while (!readers_done.load(std::memory_order_acquire)) {
                    const auto current = concurrent.snapshot(concurrent_job.id);
                    if (!current) {
                        reader_failed.store(true, std::memory_order_release);
                        return;
                    }
                    std::this_thread::yield();
                }
            });
        }
        const auto concurrent_cancel = concurrent.cancel(concurrent_job.id);
        require_test(concurrent_cancel.has_value(),
                     "concurrent cancel lost job");
        require_test(wait_terminal(concurrent, concurrent_job.id).state ==
                         product::RunLifecycleState::Cancelled,
                     "concurrent cancellation did not terminate");
        for (std::thread& waiter : event_waiters) waiter.join();
        require_test(!event_wait_failed.load(std::memory_order_acquire),
                     "concurrent event wait missed cancellation");
        readers_done.store(true, std::memory_order_release);
        for (std::thread& reader : readers) reader.join();
        require_test(!reader_failed.load(std::memory_order_acquire),
                     "concurrent snapshot lost job");

        FakeExecutor shutdown_fake;
        shutdown_fake.block("shutdown-active");
        application::JobManager shutdown_manager(
            root / "shutdown-runs", shutdown_fake.callback(), 8);
        const auto shutdown_active = shutdown_manager.submit(
            fake_model(), options("shutdown-active"));
        wait_running(shutdown_manager, shutdown_active.id);
        const auto shutdown_queued = shutdown_manager.submit(
            fake_model(), options("shutdown-queued"));
        shutdown_manager.shutdown();
        require_test(shutdown_manager.snapshot(shutdown_active.id)->state ==
                         product::RunLifecycleState::Cancelled &&
                         shutdown_manager.snapshot(shutdown_queued.id)->state ==
                         product::RunLifecycleState::Cancelled &&
                         !shutdown_fake.called("shutdown-queued"),
                     "shutdown did not cancel active/queued jobs cleanly");

        manager.shutdown();
        concurrent.shutdown();
        fs::remove_all(root);
        std::cout << "bounded process-memory JobManager tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << "bounded process-memory JobManager tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
