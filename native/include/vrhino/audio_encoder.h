#pragma once

#include <map>
#include <string>

#include "vrhino/architecture.h"
#include "vrhino/json.h"

namespace vrhino {

struct AudioEncoderComponentResult {
    std::map<std::string, Tensor> outputs;
};

// Model-name-free executor for a declarative Conv1D + pre-norm Transformer
// audio encoder graph. Conv1D is lowered exactly to the existing Conv2D
// primitive with a singleton spatial dimension; Backend never sees an audio
// encoder or checkpoint identity.
class AudioEncoderComponentExecutor {
public:
    AudioEncoderComponentExecutor(Backend& backend, WeightMap weights)
        : backend_(backend), weights_(std::move(weights)) {}

    AudioEncoderComponentResult execute(const Json& graph,
                                        const Tensor& log_mel_features);

private:
    Backend& backend_;
    WeightMap weights_;
};

}  // namespace vrhino
