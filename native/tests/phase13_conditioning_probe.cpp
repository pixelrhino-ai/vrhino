#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "phase13_safetensors.h"
#if VRHINO_USE_METAL_BACKEND
#include "vrhino/backend/metal_backend.h"
#else
#include "vrhino/backend/cuda_backend.h"
#endif
#include "vrhino/bundle.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/tensor_util.h"
#include "vrhino/tokenizer.h"

namespace {

#if VRHINO_USE_METAL_BACKEND
using NativeBackend = vrhino::MetalBackend;
#else
using NativeBackend = vrhino::CudaBackend;
#endif

std::string read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open asset: " + path);
    return std::string(std::istreambuf_iterator<char>(stream), {});
}

std::string hunyuan_video_template(const std::string& prompt) {
    return "<|start_header_id|>system<|end_header_id|>\n\n"
           "Describe the video by detailing the following aspects: "
           "1. The main content and theme of the video."
           "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
           "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
           "4. background environment, light, style and atmosphere."
           "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>";
}

vrhino::TokenizerSpec tokenizer_spec(const std::string& family) {
    vrhino::TokenizerSpec spec;
    if (family == "wan") { spec.max_length=512; spec.pad_id=0; spec.suffix_ids={1}; }
    else if (family == "llama") { spec.max_length=351; spec.pad_id=128258; spec.prefix_ids={128000}; }
    else if (family == "clip") { spec.max_length=77; spec.pad_id=49407; spec.prefix_ids={49406}; spec.suffix_ids={49407}; }
    else if (family == "ltx") { spec.format=vrhino::TokenizerAssetFormat::SentencePiece; spec.max_length=128; spec.pad_id=0; spec.suffix_ids={1}; spec.added_tokens={{"<extra_id_0>",32099,true,true}}; }
    else if (family == "mochi") { spec.format=vrhino::TokenizerAssetFormat::SentencePiece; spec.max_length=256; spec.pad_id=0; spec.suffix_ids={1}; spec.added_tokens={{"<extra_id_0>",32099,true,false}}; spec.suppress_metaspace_after_added_token=true; spec.empty_input=vrhino::EmptyInputPolicy::AllPadding; }
    else throw vrhino::Error("Unknown conditioning family: " + family);
    return spec;
}

vrhino::Tensor host_ids(const std::vector<int32_t>& values) {
    std::vector<int64_t> converted(values.begin(), values.end());
    return vrhino::host_i64({1, static_cast<int64_t>(values.size())}, converted);
}

vrhino::Tensor host_mask(const std::vector<uint8_t>& values) {
    return vrhino::host_bool({1, static_cast<int64_t>(values.size())}, values);
}

vrhino::Tensor host_f32(vrhino::Backend& backend, const vrhino::Tensor& tensor) {
    vrhino::Tensor fp32 = tensor.dtype() == vrhino::DType::F32 ? tensor
                                                               : backend.cast(tensor, vrhino::DType::F32);
    return fp32.device().is_host() ? fp32 : backend.copy_to_host(fp32);
}

vrhino::Tensor host_capture(vrhino::Backend& backend, const vrhino::Tensor& tensor) {
    if (tensor.dtype() == vrhino::DType::F32 || tensor.dtype() == vrhino::DType::F16 ||
        tensor.dtype() == vrhino::DType::BF16)
        return host_f32(backend, tensor);
    return tensor.device().is_host() ? tensor : backend.copy_to_host(tensor);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 8 || argc == 9 || argc == 10,
            "usage: phase13-conditioning-probe FAMILY GRAPH WEIGHTS_OR_INDEX TOKENIZER PROMPT OUTPUT EXEC_DTYPE [OBSERVE_BLOCK | OUTPUT_BLOCK_BEGIN OUTPUT_BLOCK_END]");
        const std::string family=argv[1], graph_path=argv[2], weight_path=argv[3],
                          tokenizer_path=argv[4], prompt=argv[5], output_path=argv[6], dtype=argv[7];
        vrhino::ConditioningObservationRequest observation;
        const vrhino::ConditioningObservationRequest* observation_pointer = nullptr;
        if (argc == 9) {
            observation.block_index = std::stoll(argv[8]);
            observation_pointer = &observation;
        } else if (argc == 10) {
            const int64_t begin = std::stoll(argv[8]);
            const int64_t end = std::stoll(argv[9]);
            vrhino::require(begin >= 0 && end >= begin,
                            "Invalid conditioning output observation block range");
            for (int64_t block = begin; block <= end; ++block)
                observation.output_block_indices.push_back(block);
            observation_pointer = &observation;
        }
        auto assets = weight_path.ends_with(".index.json")
            ? vrhino::phase13::SafeTensorSet::indexed(weight_path)
            : vrhino::phase13::SafeTensorSet::single(weight_path);
        vrhino::NativeTokenizer tokenizer(tokenizer_spec(family), read(tokenizer_path));
        const std::string token_text = family == "llama" ? hunyuan_video_template(prompt) : prompt;
        const vrhino::TokenizedInput tokenized = tokenizer.encode(token_text);
        vrhino::Tensor ids = host_ids(tokenized.input_ids), mask = host_mask(tokenized.attention_mask);

        NativeBackend backend;
#if VRHINO_USE_METAL_BACKEND
        vrhino::require(dtype == "fp32", "Metal Phase 13 conditioning probe supports FP32 only");
#endif
        backend.set_execution_dtype(dtype == "bf16" ? vrhino::DType::BF16 : vrhino::DType::F32);
        vrhino::MemoryBudget budget{21ULL<<30, 2ULL<<30, 64ULL<<30, 2ULL<<30, 1ULL<<30};
        vrhino::MemoryRuntimeOptions options; options.enabled=true; options.host_staging=false; options.prefetch=false;
        backend.configure_memory_runtime(budget, options);
        backend.set_vrm_mapped_bytes(assets.mapped_bytes());
        backend.enable_profiling(true);
        const vrhino::Json graph = vrhino::Json::parse(read(graph_path));
        vrhino::ConditioningComponentExecutor executor(backend, assets.weights());
        auto result = executor.execute(graph, ids, mask, observation_pointer);
        backend.synchronize();

        vrhino::TensorBundle output;
        output.emplace("input_ids", ids); output.emplace("attention_mask", mask);
        if (!observation.output_block_indices.empty()) {
            const auto embedding = result.captures.find("embedding");
            if (embedding != result.captures.end())
                output.emplace("capture.embedding", host_capture(backend, embedding->second));
            for (const int64_t block : observation.output_block_indices) {
                const std::string key = "block" + std::to_string(block) + ".output";
                output.emplace("capture." + key, host_capture(backend, result.captures.at(key)));
            }
        } else if (observation_pointer) {
            const std::string prefix = "block" + std::to_string(observation.block_index) + ".";
            for (const auto& [name, tensor] : result.captures)
                if (name.starts_with(prefix))
                    output.emplace("capture." + name, host_capture(backend, tensor));
        } else {
            output.emplace("hidden_states", host_f32(backend, result.hidden_states));
            if (result.pooled_output.defined()) output.emplace("pooled_output", host_f32(backend, result.pooled_output));
            for (const auto& [name, tensor] : result.captures)
                output.emplace("capture." + name, host_f32(backend, tensor));
        }
        vrhino::write_bundle(output_path, output);
        const auto memory = backend.memory_runtime_stats();
        std::cout << "status=PASS,family=" << family << ",valid_length=" << tokenized.valid_length
                  << ",mapped_bytes=" << assets.mapped_bytes()
                  << ",peak_device_bytes=" << backend.peak_device_bytes()
                  << ",weight_upload_bytes=" << backend.weight_upload_bytes()
                  << ",cache_hits=" << backend.weight_cache_hits()
                  << ",cache_misses=" << backend.weight_cache_misses()
                  << ",evictions=" << memory.evictions
                  << ",forced_syncs=" << memory.forced_syncs
                  << ",resident_weight_bytes=" << memory.accounting.device_resident_weight_bytes
                  << "\n";
        for (const auto& [name, stat] : backend.profile_stats())
            std::cout << "profile=" << name << ",calls=" << stat.calls
                      << ",device_ms=" << stat.device_milliseconds << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
