#include "vrhino/product/run_session.h"

#include <stdexcept>
#include <utility>

namespace vrhino::product {

namespace {

std::string terminal_message(const RunLifecycleState state) {
    switch (state) {
        case RunLifecycleState::Succeeded: return "Run succeeded";
        case RunLifecycleState::Failed: return "Run failed";
        case RunLifecycleState::Cancelled: return "Run cancelled";
        case RunLifecycleState::Queued:
        case RunLifecycleState::Running: break;
    }
    return {};
}

}  // namespace

RunSession::RunSession(ResolvedRunnableModel model, RunOptions options,
                       RunExecutor executor, RunEventSink event_sink)
    : model_(std::move(model)),
      options_(std::move(options)),
      executor_(std::move(executor)),
      event_sink_(std::move(event_sink)) {
    if (!executor_) throw std::invalid_argument("RunSession executor is required");
    const ProductInputSchema* schema = model_.manifest.product.input_schema
        ? &*model_.manifest.product.input_schema : nullptr;
    effective_output_ = resolve_product_output(schema, options_.output.string());
    options_.output = effective_output_;
}

bool RunSession::terminal(const RunLifecycleState state) noexcept {
    return state == RunLifecycleState::Succeeded ||
           state == RunLifecycleState::Failed ||
           state == RunLifecycleState::Cancelled;
}

RunEvent RunSession::append_event_locked(RunEvent event) {
    event.sequence = next_sequence_++;
    if (event.stage) stage_ = event.stage;
    if (event.kind == RunEventKind::Progress) {
        completed_ = event.completed;
        total_ = event.total;
        unit_ = event.unit;
    } else if (event.kind == RunEventKind::Stage) {
        completed_.reset();
        total_.reset();
        unit_.reset();
    }
    if (events_.size() == kEventHistoryCapacity) events_.pop_front();
    events_.push_back(event);
    return event;
}

void RunSession::deliver_event(const RunEvent& event) const noexcept {
    if (!event_sink_) return;
    try {
        event_sink_(event);
    } catch (...) {
        // Observers are advisory. Transport/UI failures must not alter Product
        // execution or its terminal classification.
    }
}

void RunSession::emit_executor_event(const RunEvent& event) {
    // Lifecycle belongs to RunSession; executors report only Product stages,
    // progress, and diagnostics.
    if (event.kind == RunEventKind::State) return;
    RunEvent retained;
    {
        std::lock_guard lock(mutex_);
        if (terminal(state_)) return;
        retained = append_event_locked(event);
    }
    deliver_event(retained);
}

void RunSession::finish(const RunLifecycleState state,
                        std::optional<RunFailure> failure,
                        std::optional<RunResult> result) {
    RunEvent terminal_event;
    {
        std::lock_guard lock(mutex_);
        if (terminal(state_)) return;
        if (state_ != RunLifecycleState::Running)
            throw std::logic_error("invalid RunSession terminal transition");
        state_ = state;
        failure_ = std::move(failure);
        result_ = std::move(result);
        terminal_event = append_event_locked(
            RunEvent::state_changed(state, terminal_message(state)));
    }
    deliver_event(terminal_event);
}

RunResult RunSession::run() {
    RunEvent running_event;
    {
        std::lock_guard lock(mutex_);
        if (run_invoked_)
            throw std::logic_error("RunSession::run() may be invoked only once");
        run_invoked_ = true;
        if (state_ == RunLifecycleState::Cancelled)
            throw ModelPackageError(ModelPackageErrorCode::Cancelled,
                                    "run cancelled before execution");
        if (state_ != RunLifecycleState::Queued)
            throw std::logic_error("RunSession is not queued");
        state_ = RunLifecycleState::Running;
        running_event = append_event_locked(RunEvent::state_changed(
            RunLifecycleState::Running, "Run started"));
    }
    deliver_event(running_event);

    RunOptions execution_options = options_;
    const std::function<bool()> caller_cancellation =
        execution_options.cancellation_requested;
    execution_options.cancellation_requested = [this, caller_cancellation] {
        return cancel_requested_.load(std::memory_order_acquire) ||
               (caller_cancellation && caller_cancellation());
    };

    try {
        RunResult result = executor_(
            model_, execution_options,
            [this](const RunEvent& event) { emit_executor_event(event); });
        finish(RunLifecycleState::Succeeded, std::nullopt, result);
        return result;
    } catch (const ModelPackageError& error) {
        const RunLifecycleState state =
            error.code() == ModelPackageErrorCode::Cancelled
                ? RunLifecycleState::Cancelled : RunLifecycleState::Failed;
        finish(state, RunFailure{error.code(), error.what()}, std::nullopt);
        throw;
    } catch (const std::exception& error) {
        finish(RunLifecycleState::Failed,
               RunFailure{std::nullopt, error.what()}, std::nullopt);
        throw;
    } catch (...) {
        finish(RunLifecycleState::Failed,
               RunFailure{std::nullopt, "unknown execution failure"},
               std::nullopt);
        throw;
    }
}

void RunSession::request_cancel() {
    std::optional<RunEvent> terminal_event;
    {
        std::lock_guard lock(mutex_);
        if (terminal(state_)) return;
        cancel_requested_.store(true, std::memory_order_release);
        if (state_ == RunLifecycleState::Queued) {
            state_ = RunLifecycleState::Cancelled;
            failure_ = RunFailure{ModelPackageErrorCode::Cancelled,
                                  "CANCELLED: run cancelled before execution"};
            terminal_event = append_event_locked(RunEvent::state_changed(
                RunLifecycleState::Cancelled, "Run cancelled"));
        }
    }
    if (terminal_event) deliver_event(*terminal_event);
}

bool RunSession::cancel_requested() const noexcept {
    return cancel_requested_.load(std::memory_order_acquire);
}

RunSessionSnapshot RunSession::snapshot() const {
    std::lock_guard lock(mutex_);
    RunSessionSnapshot result;
    result.state = state_;
    result.cancel_requested =
        cancel_requested_.load(std::memory_order_acquire);
    result.stage = stage_;
    result.completed = completed_;
    result.total = total_;
    result.unit = unit_;
    result.result = result_;
    result.failure = failure_;
    result.latest_event_sequence = next_sequence_ - 1;
    return result;
}

RunEventHistorySnapshot RunSession::events_after(const uint64_t sequence) const {
    std::lock_guard lock(mutex_);
    RunEventHistorySnapshot result;
    result.latest_sequence = next_sequence_ - 1;
    if (events_.empty()) return result;
    result.first_retained_sequence = events_.front().sequence;
    result.history_truncated = sequence < events_.front().sequence - 1;
    for (const RunEvent& event : events_)
        if (event.sequence > sequence) result.events.push_back(event);
    return result;
}

std::filesystem::path RunSession::effective_output_path() const {
    return effective_output_;
}

}  // namespace vrhino::product
