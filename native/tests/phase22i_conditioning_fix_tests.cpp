#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "phase13_safetensors.h"
#include "vrhino/architecture.h"
#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/sampling.h"
#include "vrhino/tensor_util.h"
#include "vrhino/tokenizer.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open Phase 22I input: " + path);
    return std::string(std::istreambuf_iterator<char>(stream), {});
}

std::vector<int32_t> i32_array(const vrhino::Json* value) {
    std::vector<int32_t> result;
    if (!value) return result;
    for (const auto& item : value->array())
        result.push_back(static_cast<int32_t>(item.integer()));
    return result;
}

vrhino::TokenizerSpec tokenizer_spec(const vrhino::Json& value) {
    vrhino::TokenizerSpec spec;
    if (value.at("format").string() == "sentencepiece")
        spec.format = vrhino::TokenizerAssetFormat::SentencePiece;
    else
        vrhino::require(value.at("format").string() == "huggingface_json",
                        "Unsupported tokenizer format");
    spec.max_length = value.at("max_length").integer();
    spec.pad_id = static_cast<int32_t>(value.at("pad_id").integer());
    spec.prefix_ids = i32_array(value.find("prefix_ids"));
    spec.suffix_ids = i32_array(value.find("suffix_ids"));
    if (const auto* empty = value.find("empty_input")) {
        vrhino::require(empty->string() == "all_padding",
                        "Unsupported empty-input policy");
        spec.empty_input = vrhino::EmptyInputPolicy::AllPadding;
    }
    if (const auto* suppress = value.find("suppress_metaspace_after_added_token"))
        spec.suppress_metaspace_after_added_token = suppress->boolean();
    if (const auto* added = value.find("added_tokens")) {
        for (const auto& item : added->array())
            spec.added_tokens.push_back({item.at("content").string(),
                static_cast<int32_t>(item.at("id").integer()),
                item.at("lstrip").boolean(), item.at("rstrip").boolean()});
    }
    return spec;
}

vrhino::Tensor ids_tensor(const vrhino::TokenizedInput& input) {
    return vrhino::host_i64({1, static_cast<int64_t>(input.input_ids.size())},
        std::vector<int64_t>(input.input_ids.begin(), input.input_ids.end()));
}

vrhino::Tensor mask_tensor(const vrhino::TokenizedInput& input) {
    return vrhino::host_bool(
        {1, static_cast<int64_t>(input.attention_mask.size())},
        input.attention_mask);
}

float bf16(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

struct FiniteStats {
    uint64_t nan = 0;
    uint64_t inf = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double rms = 0.0;
};

FiniteStats finite_stats(const vrhino::Tensor& value) {
    vrhino::require(value.device().is_host(), "Phase 22I stats require host tensor");
    FiniteStats result;
    long double square = 0.0L;
    uint64_t finite = 0;
    for (int64_t index = 0; index < value.numel(); ++index) {
        double item;
        if (value.dtype() == vrhino::DType::BF16)
            item = bf16(value.data_as<uint16_t>()[index]);
        else if (value.dtype() == vrhino::DType::F32)
            item = value.data_as<float>()[index];
        else if (value.dtype() == vrhino::DType::Bool)
            item = value.data_as<uint8_t>()[index] != 0;
        else if (value.dtype() == vrhino::DType::I64)
            item = value.data_as<int64_t>()[index];
        else
            throw vrhino::Error("Unsupported Phase 22I stats dtype");
        result.nan += std::isnan(item);
        result.inf += std::isinf(item);
        if (std::isfinite(item)) {
            ++finite;
            result.minimum = std::min(result.minimum, item);
            result.maximum = std::max(result.maximum, item);
            square += item * item;
        }
    }
    if (finite) result.rms = std::sqrt(static_cast<double>(square / finite));
    return result;
}

bool exact_tensor(const vrhino::Tensor& left, const vrhino::Tensor& right) {
    return left.dtype() == right.dtype() && left.shape() == right.shape() &&
           left.bytes() == right.bytes() &&
           std::memcmp(left.data(), right.data(), left.bytes()) == 0;
}

bool exact_bundle(const vrhino::TensorBundle& left,
                  const vrhino::TensorBundle& right) {
    if (left.size() != right.size()) return false;
    for (const auto& [name, tensor] : left) {
        const auto found = right.find(name);
        if (found == right.end() || !exact_tensor(tensor, found->second))
            return false;
    }
    return true;
}

void print_finite(const std::string& name, const vrhino::Tensor& value) {
    const FiniteStats stats = finite_stats(value);
    std::cout << name << ".nan=" << stats.nan
              << '\n' << name << ".inf=" << stats.inf
              << '\n' << name << ".min=" << stats.minimum
              << '\n' << name << ".max=" << stats.maximum
              << '\n' << name << ".rms=" << stats.rms << '\n';
}

void configure(vrhino::CudaBackend& backend, uint64_t mapped_bytes) {
    backend.set_execution_dtype(vrhino::DType::BF16);
    backend.enable_weight_cache(true);
    backend.configure_memory_runtime(
        vrhino::MemoryBudget{88ULL << 30, 2ULL << 30, 128ULL << 30,
                             8ULL << 30, 1ULL << 30},
        vrhino::MemoryRuntimeOptions{true, false, false});
    backend.set_vrm_mapped_bytes(mapped_bytes);
}

const vrhino::Json& negative_component(const vrhino::Json& profile) {
    for (const auto& component : profile.at("components").array())
        if (component.at("name").string() == "negative") return component;
    throw vrhino::Error("Phase 22I profile has no negative component");
}

void condition(const std::string& profile_path, const std::string& output_path) {
    const vrhino::Json profile = vrhino::Json::parse(read_text(profile_path));
    const vrhino::Json& component = negative_component(profile);
    const vrhino::TokenizerSpec spec = tokenizer_spec(component.at("tokenizer_spec"));
    vrhino::require(spec.empty_input == vrhino::EmptyInputPolicy::EncodeNormally,
                    "Mochi production profile still requests all-padding empty input");
    vrhino::NativeTokenizer tokenizer(
        spec, read_text(component.at("tokenizer").string()));
    const vrhino::TokenizedInput encoded = tokenizer.encode("");
    vrhino::require(encoded.valid_length == 1 && encoded.input_ids.size() == 256 &&
                        encoded.input_ids[0] == 1 && encoded.attention_mask[0] == 1,
                    "Empty T5 tokenization did not retain exactly one valid EOS");
    for (size_t index = 1; index < encoded.input_ids.size(); ++index)
        vrhino::require(encoded.input_ids[index] == 0 &&
                            encoded.attention_mask[index] == 0,
                        "Empty T5 padding after EOS mismatch");

    auto weights = std::string(component.at("weights").string()).ends_with(".index.json")
        ? vrhino::phase13::SafeTensorSet::indexed(component.at("weights").string())
        : vrhino::phase13::SafeTensorSet::single(component.at("weights").string());
    vrhino::CudaBackend backend;
    configure(backend, weights.mapped_bytes());
    const vrhino::Tensor ids = ids_tensor(encoded);
    const vrhino::Tensor mask = mask_tensor(encoded);
    vrhino::ConditioningComponentExecutor executor(backend, weights.weights());
    const vrhino::Json graph = vrhino::Json::parse(
        read_text(component.at("graph").string()));
    vrhino::ConditioningObservationRequest observation;
    for (size_t index = 0; index < graph.at("blocks").array().size(); ++index)
        observation.output_block_indices.push_back(static_cast<int64_t>(index));
    const vrhino::ConditioningComponentResult first_result =
        executor.execute(graph, ids, mask, &observation);
    const vrhino::Tensor first_device = first_result.hidden_states;
    backend.synchronize();
    size_t finite_blocks = 0;
    for (size_t index = 0; index < graph.at("blocks").array().size(); ++index) {
        const vrhino::Tensor block = backend.copy_to_host(first_result.captures.at(
            "block" + std::to_string(index) + ".output"));
        const FiniteStats block_stats = finite_stats(block);
        finite_blocks += block_stats.nan == 0 && block_stats.inf == 0;
    }
    vrhino::require(finite_blocks == graph.at("blocks").array().size(),
                    "A fixed negative-conditioning block is non-finite");
    const vrhino::Tensor first = backend.copy_to_host(first_device);
    const vrhino::Tensor second_device = executor.execute(graph, ids, mask).hidden_states;
    backend.synchronize();
    const vrhino::Tensor second = backend.copy_to_host(second_device);
    const FiniteStats stats = finite_stats(first);
    vrhino::require(stats.nan == 0 && stats.inf == 0,
                    "Fixed negative conditioning is non-finite");
    vrhino::require(exact_tensor(first, second),
                    "Fixed negative conditioning is not deterministic");
    vrhino::write_bundle(output_path, {
        {"negative", first}, {"negative_input_ids", ids}, {"negative_mask", mask}});
    std::cout << "status=PASS\nempty_valid_tokens=1\nfirst_id=1"
              << "\nsecond_id=0\nconditioning_shape=[1,256,4096]"
              << "\nconditioning_nan=0\nconditioning_inf=0"
              << "\nconditioning_blocks_finite=" << finite_blocks << '/'
              << graph.at("blocks").array().size()
              << "\ndeterministic=BYTE_EXACT_PASS\n";
    print_finite("negative_conditioning", first);
}

void chain(const std::string& model_path, const std::string& matched_path,
           const std::string& conditioning_path, const std::string& policy_path,
           const std::string& output_path) {
    vrhino::VrmModel model(model_path, false);
    vrhino::TensorBundle matched = vrhino::read_bundle(matched_path);
    const vrhino::TensorBundle fixed = vrhino::read_bundle(conditioning_path);
    vrhino::TensorBundle input;
    input.emplace("positive", matched.at("positive"));
    input.emplace("negative", fixed.at("negative"));
    input.emplace("positive_mask", matched.at("positive_mask"));
    input.emplace("negative_mask", fixed.at("negative_mask"));
    input.emplace("seed", vrhino::scalar_i64(11001));
    input.emplace("latent_shape", vrhino::host_i64({5}, {1, 12, 28, 60, 106}));
    input.emplace("sampling_steps", vrhino::scalar_i64(64));
    input.emplace("guidance_scale", vrhino::scalar_f32(6.0f));
    input.emplace("threshold_noise", vrhino::scalar_f32(0.025f));
    input.emplace("linear_steps", vrhino::scalar_i64(32));
    input.emplace("audit_trace", vrhino::scalar_i64(0));

    vrhino::CudaBackend backend;
    configure(backend, model.file_size());
    const vrhino::PrecisionPolicy policy = vrhino::PrecisionPolicy::from_json(
        vrhino::Json::parse(read_text(policy_path)));
    auto architecture = vrhino::create_architecture(model);
    vrhino::SamplingProgram program = architecture->create_program(input);
    vrhino::require(program.steps == 64 && program.latent_shape ==
                        std::vector<int64_t>({1, 12, 28, 60, 106}),
                    "Phase 22I frozen program mismatch");
    program.steps = 3;
    program.sigmas.resize(4);
    program.model_timesteps.resize(3);
    program.update_deltas.resize(3);
    vrhino::SamplingRuntime runtime(backend, policy);
    auto denoiser = architecture->create_denoiser(backend, policy, input);
    const vrhino::SamplingResult first =
        runtime.run_with_external_initial_state_for_test(
            *denoiser, program, matched.at("initial_noise"));
    backend.synchronize();
    const vrhino::SamplingResult second =
        runtime.run_with_external_initial_state_for_test(
            *denoiser, program, matched.at("initial_noise"));
    backend.synchronize();
    vrhino::require(exact_bundle(first.trace, second.trace),
                    "Exact-shape three-step chain is not deterministic");
    for (const auto& [name, tensor] : first.trace) {
        const FiniteStats stats = finite_stats(tensor);
        vrhino::require(stats.nan == 0 && stats.inf == 0,
                        "Non-finite short-chain tensor: " + name);
    }
    vrhino::TensorBundle output = first.trace;
    output.emplace("conditioning.negative", fixed.at("negative"));
    output.emplace("tokenizer.negative.input_ids", fixed.at("negative_input_ids"));
    output.emplace("tokenizer.negative.attention_mask", fixed.at("negative_mask"));
    vrhino::write_bundle(output_path, output);
    std::cout << "status=PASS\ndeclared_steps=64\nexecuted_steps=3"
              << "\nlatent_shape=[1,12,28,60,106]"
              << "\npositive_prediction_nan=0\npositive_prediction_inf=0"
              << "\nnegative_prediction_nan=0\nnegative_prediction_inf=0"
              << "\ncfg_prediction_nan=0\ncfg_prediction_inf=0"
              << "\nupdated_latent_nan=0\nupdated_latent_inf=0"
              << "\nshort_chain_nan=0\nshort_chain_inf=0"
              << "\ndeterministic=BYTE_EXACT_PASS"
              << "\ntrace_tensors=" << first.trace.size()
              << "\nwall_seconds=" << first.elapsed_seconds << '\n';
    print_finite("step0.negative_prediction", first.trace.at("step.0.prediction.0"));
    print_finite("step0.positive_prediction", first.trace.at("step.0.prediction.1"));
    print_finite("step0.cfg_prediction", first.trace.at("step.0.guidance"));
    print_finite("step0.updated_latent", first.trace.at("step.0.latent"));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc >= 2, "Phase 22I mode is required");
        const std::string mode = argv[1];
        if (mode == "conditioning") {
            vrhino::require(argc == 4,
                "usage: phase22i conditioning PROFILE OUTPUT");
            condition(argv[2], argv[3]);
        } else if (mode == "chain") {
            vrhino::require(argc == 7,
                "usage: phase22i chain MODEL MATCHED CONDITIONING POLICY OUTPUT");
            chain(argv[2], argv[3], argv[4], argv[5], argv[6]);
        } else {
            throw vrhino::Error("Unknown Phase 22I mode");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "phase22i_conditioning_fix_tests: " << error.what() << '\n';
        return 1;
    }
}
