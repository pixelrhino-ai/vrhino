#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <pthread.h>
#include <unistd.h>

#include "vrhino/product/run_session.h"

namespace product = vrhino::product;

namespace {

volatile std::sig_atomic_t cli_interrupt_requested = 0;

void cli_interrupt_handler(int) { cli_interrupt_requested = 1; }

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

product::ResolvedRunnableModel fake_model() {
    product::ResolvedRunnableModel model;
    model.manifest.identity.name_space = "vrhino";
    model.manifest.identity.name = "fake";
    model.manifest.identity.version = "1.0.0";
    model.manifest.product.family = "text_to_video";
    return model;
}

product::RunResult fake_result() {
    product::RunResult result;
    result.identity = fake_model().manifest.identity;
    result.preset = "default";
    result.seed = 5703;
    result.width = 64;
    result.height = 64;
    result.frames = 3;
    result.fps = 25;
    return result;
}

product::RunOptions fake_options() {
    product::RunOptions options;
    options.model_reference = "vrhino/fake:1.0.0";
    options.prompt = "cpu-only fake";
    return options;
}

template <typename Function>
product::ModelPackageError expect_package_error(
        const product::ModelPackageErrorCode code, Function&& function,
        const std::string& context) {
    try {
        function();
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == code, context + ": wrong error code");
        return error;
    }
    throw std::runtime_error(context + ": expected ModelPackageError");
}

product::RunExecutor successful_executor() {
    return [](const product::ResolvedRunnableModel&,
              const product::RunOptions&,
              product::RunEventSink events) {
        if (events) {
            events(product::RunEvent::stage_changed(
                product::RunStage::Loading, "Loading model"));
            events(product::RunEvent::stage_changed(
                product::RunStage::Sampling, "Sampling"));
            for (uint64_t step = 1; step <= 3; ++step)
                events(product::RunEvent::progress(
                    product::RunStage::Sampling, step, 3,
                    product::RunProgressUnit::Step,
                    "Sampling " + std::to_string(step) + "/3"));
            events(product::RunEvent::stage_changed(
                product::RunStage::Finalizing, "Done"));
        }
        return fake_result();
    };
}

void require_strict_sequence(const std::vector<product::RunEvent>& events,
                             const std::string& context) {
    for (size_t index = 0; index < events.size(); ++index) {
        require_test(events[index].sequence == index + events.front().sequence,
                     context + ": non-contiguous event sequence");
        if (index != 0)
            require_test(events[index - 1].sequence < events[index].sequence,
                         context + ": non-increasing event sequence");
    }
}

void test_success_and_event_order() {
    std::vector<product::RunEvent> observed;
    product::RunSession session(
        fake_model(), fake_options(), successful_executor(),
        [&](const product::RunEvent& event) { observed.push_back(event); });

    require_test(session.snapshot().state == product::RunLifecycleState::Queued,
                 "new session is not queued");
    require_test(session.effective_output_path() == "output.mp4",
                 "canonical default output was not retained");
    const product::RunResult result = session.run();
    require_test(result.seed == 5703, "success result changed");
    const product::RunSessionSnapshot snapshot = session.snapshot();
    require_test(snapshot.state == product::RunLifecycleState::Succeeded,
                 "success did not reach succeeded");
    require_test(snapshot.result && snapshot.result->seed == 5703,
                 "terminal result was not retained");
    require_test(!snapshot.failure, "success retained a failure");
    require_test(snapshot.stage == product::RunStage::Finalizing,
                 "latest stage was not retained");
    require_test(observed.size() == 8, "unexpected success event count");
    require_strict_sequence(observed, "success events");
    require_test(observed.front().sequence == 1,
                 "session event sequence did not start at one");
    require_test(observed.front().kind == product::RunEventKind::State &&
                     observed.front().state ==
                         product::RunLifecycleState::Running,
                 "first event was not running state");
    require_test(observed[2].kind == product::RunEventKind::Stage &&
                     observed[2].stage == product::RunStage::Sampling,
                 "sampling stage order changed");
    require_test(observed[3].completed == 1 && observed[5].completed == 3 &&
                     observed[5].total == 3 &&
                     observed[5].unit == product::RunProgressUnit::Step,
                 "structured sampling progress changed");
    require_test(observed.back().kind == product::RunEventKind::State &&
                     observed.back().state ==
                         product::RunLifecycleState::Succeeded,
                 "last event was not succeeded state");

    session.request_cancel();
    session.request_cancel();
    require_test(session.snapshot().state == product::RunLifecycleState::Succeeded,
                 "cancel-after-success mutated terminal state");
    require_test(!session.snapshot().cancel_requested,
                 "cancel-after-success mutated cancellation state");
    require_test(session.events_after(0).events.size() == observed.size(),
                 "cancel-after-success emitted an event");

    try {
        (void)session.run();
        throw std::runtime_error("second run invocation unexpectedly succeeded");
    } catch (const std::logic_error&) {
    }
    require_test(session.snapshot().state == product::RunLifecycleState::Succeeded,
                 "second run invocation mutated terminal state");
}

void test_failure() {
    product::RunSession session(
        fake_model(), fake_options(),
        [](const product::ResolvedRunnableModel&, const product::RunOptions&,
           product::RunEventSink events) -> product::RunResult {
            events(product::RunEvent::stage_changed(
                product::RunStage::Validation, "Validating"));
            throw product::ModelPackageError(
                product::ModelPackageErrorCode::RuntimeError,
                "safe fake execution failure");
        });
    expect_package_error(product::ModelPackageErrorCode::RuntimeError,
                         [&] { (void)session.run(); }, "failure session");
    const product::RunSessionSnapshot snapshot = session.snapshot();
    require_test(snapshot.state == product::RunLifecycleState::Failed,
                 "failure did not reach failed");
    require_test(snapshot.failure && snapshot.failure->code ==
                     product::ModelPackageErrorCode::RuntimeError &&
                     snapshot.failure->message ==
                         "RUNTIME_ERROR: safe fake execution failure",
                 "failure detail was not retained");
    require_test(!snapshot.result, "failure retained a result");
    require_test(!snapshot.cancel_requested,
                 "failure was classified as cancellation");
    session.request_cancel();
    require_test(session.snapshot().state == product::RunLifecycleState::Failed,
                 "cancel-after-failure mutated terminal state");
}

void test_cancel_before_start() {
    std::atomic<int> invocations{0};
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("vrhino-run-session-cancel-before-" +
         std::to_string(static_cast<long long>(::getpid())));
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    product::RunOptions options = fake_options();
    options.output = root / "output.mp4";
    product::RunSession session(
        fake_model(), options,
        [&](const product::ResolvedRunnableModel&, const product::RunOptions&,
            product::RunEventSink) {
            ++invocations;
            return fake_result();
        });
    session.request_cancel();
    session.request_cancel();
    require_test(session.snapshot().state == product::RunLifecycleState::Cancelled,
                 "pre-start cancellation did not become terminal");
    require_test(session.cancel_requested(),
                 "pre-start cancellation flag was not retained");
    expect_package_error(product::ModelPackageErrorCode::Cancelled,
                         [&] { (void)session.run(); }, "cancel before start");
    require_test(invocations.load() == 0,
                 "cancel-before-start invoked Product executor");
    std::filesystem::path partial = options.output;
    partial += ".partial";
    require_test(!std::filesystem::exists(options.output) &&
                     !std::filesystem::exists(partial) &&
                     !std::filesystem::exists(root),
                 "cancel-before-start created output state");
    const auto history = session.events_after(0);
    require_test(history.events.size() == 1 &&
                     history.events.front().sequence == 1 &&
                     history.events.front().state ==
                         product::RunLifecycleState::Cancelled,
                 "cancel-before-start emitted inconsistent events");
    session.request_cancel();
    require_test(session.events_after(0).events.size() == 1,
                 "cancel-after-cancel emitted duplicate terminal event");
}

void test_direct_cross_thread_cancellation_and_second_session() {
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    std::atomic<bool> predicate_observed{false};
    std::optional<product::ModelPackageErrorCode> thread_error;

    product::RunSession session(
        fake_model(), fake_options(),
        [&](const product::ResolvedRunnableModel&,
            const product::RunOptions& options,
            product::RunEventSink events) -> product::RunResult {
            {
                std::lock_guard lock(mutex);
                entered = true;
            }
            condition.notify_one();
            events(product::RunEvent::stage_changed(
                product::RunStage::Sampling, "Cooperative fake execution"));
            while (!options.cancellation_requested()) std::this_thread::yield();
            predicate_observed.store(true);
            throw product::ModelPackageError(
                product::ModelPackageErrorCode::Cancelled,
                "fake executor observed cancellation");
        });

    std::thread runner([&] {
        try {
            (void)session.run();
        } catch (const product::ModelPackageError& error) {
            thread_error = error.code();
        }
    });
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return entered; });
    }
    require_test(session.snapshot().state == product::RunLifecycleState::Running,
                 "cross-thread test did not observe running");
    session.request_cancel();
    session.request_cancel();
    runner.join();
    require_test(predicate_observed.load(),
                 "executor did not observe session cancellation predicate");
    require_test(thread_error == product::ModelPackageErrorCode::Cancelled,
                 "cross-thread cancellation did not preserve error code");
    require_test(session.snapshot().state == product::RunLifecycleState::Cancelled,
                 "cross-thread cancellation did not reach cancelled");
    require_test(session.events_after(0).events.back().state ==
                     product::RunLifecycleState::Cancelled,
                 "cross-thread cancellation missed terminal event");

    product::RunSession second(
        fake_model(), fake_options(), successful_executor());
    (void)second.run();
    require_test(second.snapshot().state == product::RunLifecycleState::Succeeded &&
                     !second.cancel_requested(),
                 "cancelled session leaked into independent session");
}

void test_concurrent_snapshots() {
    std::atomic<bool> executor_done{false};
    std::atomic<bool> reader_stop{false};
    std::atomic<bool> snapshots_ordered{true};
    std::atomic<uint64_t> reads{0};
    product::RunSession session(
        fake_model(), fake_options(),
        [&](const product::ResolvedRunnableModel&, const product::RunOptions&,
            product::RunEventSink events) {
            for (uint64_t frame = 1; frame <= 2000; ++frame) {
                events(product::RunEvent::progress(
                    product::RunStage::Composite, frame, 2000,
                    product::RunProgressUnit::Frame));
                if ((frame % 16) == 0) std::this_thread::yield();
            }
            executor_done.store(true);
            return fake_result();
        });

    std::thread runner([&] { (void)session.run(); });
    std::thread reader([&] {
        while (!reader_stop.load()) {
            const product::RunSessionSnapshot state = session.snapshot();
            const product::RunEventHistorySnapshot events =
                session.events_after(state.latest_event_sequence > 8
                                         ? state.latest_event_sequence - 8 : 0);
            uint64_t previous = 0;
            for (const product::RunEvent& event : events.events) {
                if (event.sequence <= previous) snapshots_ordered.store(false);
                previous = event.sequence;
            }
            ++reads;
            if (executor_done.load() &&
                state.state == product::RunLifecycleState::Succeeded)
                break;
        }
    });
    runner.join();
    reader_stop.store(true);
    reader.join();
    require_test(reads.load() > 0, "concurrent reader did not execute");
    require_test(snapshots_ordered.load(),
                 "concurrent event snapshot is unordered");
    require_test(session.snapshot().state == product::RunLifecycleState::Succeeded,
                 "snapshot stress did not succeed");
}

void test_bounded_history() {
    constexpr uint64_t emitted = product::RunSession::kEventHistoryCapacity + 73;
    product::RunSession session(
        fake_model(), fake_options(),
        [](const product::ResolvedRunnableModel&, const product::RunOptions&,
           product::RunEventSink events) {
            for (uint64_t index = 1; index <= emitted; ++index)
                events(product::RunEvent::progress(
                    product::RunStage::Sampling, index, emitted,
                    product::RunProgressUnit::Step));
            return fake_result();
        });
    (void)session.run();
    const product::RunEventHistorySnapshot history = session.events_after(0);
    require_test(history.events.size() ==
                     product::RunSession::kEventHistoryCapacity,
                 "event history exceeded its fixed capacity");
    require_test(history.history_truncated,
                 "dropped history was not reported");
    require_test(history.latest_sequence == emitted + 2,
                 "event sequence reset or skipped terminal events");
    require_test(history.first_retained_sequence ==
                     history.latest_sequence -
                         product::RunSession::kEventHistoryCapacity + 1,
                 "bounded history retained wrong window");
    require_strict_sequence(history.events, "bounded history");
    require_test(history.events.back().state ==
                     product::RunLifecycleState::Succeeded,
                 "bounded history lost newest terminal event");

    const uint64_t after = history.latest_sequence - 4;
    const product::RunEventHistorySnapshot tail = session.events_after(after);
    require_test(!tail.history_truncated && tail.events.size() == 4,
                 "events_after did not return bounded tail");
    require_test(tail.events.front().sequence == after + 1 &&
                     tail.events.back().sequence == history.latest_sequence,
                 "events_after tail ordering changed");
}

void test_throwing_event_consumer_is_advisory() {
    product::RunSession session(
        fake_model(), fake_options(), successful_executor(),
        [](const product::RunEvent&) { throw std::runtime_error("consumer failed"); });
    (void)session.run();
    require_test(session.snapshot().state == product::RunLifecycleState::Succeeded,
                 "event consumer failure altered execution");
    require_test(session.events_after(0).events.size() == 8,
                 "event consumer failure altered retention");
}

void test_cli_signal_adapter_and_exit_mapping() {
    cli_interrupt_requested = 0;
    const auto previous = std::signal(SIGINT, cli_interrupt_handler);
    const pthread_t execution_thread = pthread_self();
    std::atomic<bool> entered{false};
    std::atomic<int> signal_delivery_status{-1};
    int exit_code = 0;
    product::RunOptions options = fake_options();
    options.cancellation_requested = [] {
        return cli_interrupt_requested != 0;
    };
    product::RunSession session(
        fake_model(), options,
        [&](const product::ResolvedRunnableModel&,
            const product::RunOptions& execution_options,
            product::RunEventSink) -> product::RunResult {
            entered.store(true, std::memory_order_release);
            while (!execution_options.cancellation_requested())
                std::this_thread::yield();
            throw product::ModelPackageError(
                product::ModelPackageErrorCode::Cancelled,
                "CLI cancellation observed");
        });
    std::thread interrupter([&] {
        while (!entered.load(std::memory_order_acquire))
            std::this_thread::yield();
        signal_delivery_status.store(
            pthread_kill(execution_thread, SIGINT), std::memory_order_release);
    });
    try {
        (void)session.run();
    } catch (const product::ModelPackageError& error) {
        exit_code = error.code() == product::ModelPackageErrorCode::Cancelled
            ? 130 : 1;
    }
    interrupter.join();
    std::signal(SIGINT, previous);
    require_test(signal_delivery_status.load(std::memory_order_acquire) == 0,
                 "could not deliver test SIGINT");
    require_test(exit_code == 130, "CLI cancellation did not map to exit 130");
    require_test(session.snapshot().state == product::RunLifecycleState::Cancelled,
                 "CLI signal cancellation did not reach cancelled");

    cli_interrupt_requested = 0;
    product::RunSession independent(
        fake_model(), fake_options(), successful_executor());
    (void)independent.run();
    require_test(independent.snapshot().state ==
                     product::RunLifecycleState::Succeeded,
                 "CLI signal state leaked into an independent session");
}

void test_explicit_output_retention() {
    product::RunOptions options = fake_options();
    options.output = std::filesystem::path("custom-output.mp4");
    product::RunSession session(
        fake_model(), options, successful_executor());
    require_test(session.effective_output_path() == options.output,
                 "explicit output path was changed");

    product::ResolvedRunnableModel schema_model = fake_model();
    product::ProductInputSchema schema;
    schema.identity = product::kProductInputSchemaV1;
    product::ProductFieldDeclaration output;
    output.name = "output";
    output.type = product::ProductValueType::MediaMp4;
    output.default_value = std::string("schema-default.mp4");
    schema.outputs.push_back(std::move(output));
    schema_model.manifest.product.input_schema = std::move(schema);
    product::RunSession schema_session(
        std::move(schema_model), fake_options(),
        [](const product::ResolvedRunnableModel&,
           const product::RunOptions& execution_options,
           product::RunEventSink) {
            require_test(execution_options.output == "schema-default.mp4",
                         "executor did not receive canonical output default");
            return fake_result();
        });
    require_test(schema_session.effective_output_path() ==
                     "schema-default.mp4",
                 "canonical Product schema output default was changed");
    (void)schema_session.run();
}

}  // namespace

int main() {
    try {
        test_success_and_event_order();
        test_failure();
        test_cancel_before_start();
        test_direct_cross_thread_cancellation_and_second_session();
        test_concurrent_snapshots();
        test_bounded_history();
        test_throwing_event_consumer_is_advisory();
        test_cli_signal_adapter_and_exit_mapping();
        test_explicit_output_retention();
        std::cout << "run-session: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "run-session: FAIL: " << error.what() << '\n';
        return 1;
    }
}
