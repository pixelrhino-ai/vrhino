#include "vrhino/product/run_event.h"

#include <utility>

namespace vrhino::product {

RunEvent RunEvent::state_changed(const RunLifecycleState value,
                                 std::string message) {
    RunEvent event;
    event.kind = RunEventKind::State;
    event.state = value;
    event.message = std::move(message);
    return event;
}

RunEvent RunEvent::stage_changed(const RunStage value, std::string message) {
    RunEvent event;
    event.kind = RunEventKind::Stage;
    event.stage = value;
    event.message = std::move(message);
    return event;
}

RunEvent RunEvent::progress(const RunStage value, const uint64_t completed,
                            const uint64_t total, const RunProgressUnit unit,
                            std::string message) {
    RunEvent event;
    event.kind = RunEventKind::Progress;
    event.stage = value;
    event.completed = completed;
    event.total = total;
    event.unit = unit;
    event.message = std::move(message);
    return event;
}

RunEvent RunEvent::diagnostic(std::string message,
                              const std::optional<RunStage> stage) {
    RunEvent event;
    event.kind = RunEventKind::Message;
    event.stage = stage;
    event.message = std::move(message);
    return event;
}

const char* run_lifecycle_state_name(const RunLifecycleState state) noexcept {
    switch (state) {
        case RunLifecycleState::Queued: return "queued";
        case RunLifecycleState::Running: return "running";
        case RunLifecycleState::Succeeded: return "succeeded";
        case RunLifecycleState::Failed: return "failed";
        case RunLifecycleState::Cancelled: return "cancelled";
    }
    return "failed";
}

const char* run_event_kind_name(const RunEventKind kind) noexcept {
    switch (kind) {
        case RunEventKind::State: return "state";
        case RunEventKind::Stage: return "stage";
        case RunEventKind::Progress: return "progress";
        case RunEventKind::Message: return "message";
    }
    return "message";
}

const char* run_stage_name(const RunStage stage) noexcept {
    switch (stage) {
        case RunStage::Admission: return "admission";
        case RunStage::Validation: return "validation";
        case RunStage::Loading: return "loading";
        case RunStage::MediaInput: return "media_input";
        case RunStage::Conditioning: return "conditioning";
        case RunStage::Analysis: return "analysis";
        case RunStage::SourcePreparation: return "source_preparation";
        case RunStage::Sampling: return "sampling";
        case RunStage::Decoding: return "decoding";
        case RunStage::Composite: return "composite";
        case RunStage::Encoding: return "encoding";
        case RunStage::Finalizing: return "finalizing";
    }
    return "finalizing";
}

const char* run_progress_unit_name(const RunProgressUnit unit) noexcept {
    switch (unit) {
        case RunProgressUnit::Step: return "step";
        case RunProgressUnit::Frame: return "frame";
        case RunProgressUnit::Chunk: return "chunk";
    }
    return "step";
}

}  // namespace vrhino::product
