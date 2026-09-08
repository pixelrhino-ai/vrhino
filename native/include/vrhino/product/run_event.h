#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace vrhino::product {

enum class RunLifecycleState {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
};

enum class RunEventKind {
    State,
    Stage,
    Progress,
    Message,
};

enum class RunStage {
    Admission,
    Validation,
    Loading,
    MediaInput,
    Conditioning,
    Analysis,
    SourcePreparation,
    Sampling,
    Decoding,
    Composite,
    Encoding,
    Finalizing,
};

enum class RunProgressUnit {
    Step,
    Frame,
    Chunk,
};

struct RunEvent {
    // Product emitters leave sequence at zero. RunSession assigns the
    // per-session sequence immediately before retaining/delivering the event.
    uint64_t sequence = 0;
    RunEventKind kind = RunEventKind::Message;
    std::optional<RunLifecycleState> state;
    std::optional<RunStage> stage;
    std::optional<uint64_t> completed;
    std::optional<uint64_t> total;
    std::optional<RunProgressUnit> unit;
    std::string message;

    static RunEvent state_changed(RunLifecycleState state,
                                  std::string message = {});
    static RunEvent stage_changed(RunStage stage, std::string message = {});
    static RunEvent progress(RunStage stage, uint64_t completed,
                             uint64_t total, RunProgressUnit unit,
                             std::string message = {});
    static RunEvent diagnostic(std::string message,
                               std::optional<RunStage> stage = std::nullopt);
};

using RunEventSink = std::function<void(const RunEvent&)>;

const char* run_lifecycle_state_name(RunLifecycleState state) noexcept;
const char* run_event_kind_name(RunEventKind kind) noexcept;
const char* run_stage_name(RunStage stage) noexcept;
const char* run_progress_unit_name(RunProgressUnit unit) noexcept;

}  // namespace vrhino::product
