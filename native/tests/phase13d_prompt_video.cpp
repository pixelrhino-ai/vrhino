#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "phase13_safetensors.h"
#include "vrhino/architecture.h"
#if VRHINO_USE_METAL_BACKEND
#include "vrhino/backend/metal_backend.h"
#else
#include "vrhino/backend/cuda_backend.h"
#endif
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/input.h"
#include "vrhino/json.h"
#include "vrhino/loader.h"
#include "vrhino/precision.h"
#include "vrhino/runtime.h"
#include "vrhino/tensor_util.h"
#include "vrhino/tokenizer.h"

namespace {

#if VRHINO_USE_METAL_BACKEND
using NativeBackend = vrhino::MetalBackend;
constexpr const char* kBackendName = "metal";
#else
using NativeBackend = vrhino::CudaBackend;
constexpr const char* kBackendName = "cuda";
#endif

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    vrhino::require(stream.good(), "Cannot open asset: " + path);
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
    else vrhino::require(value.at("format").string() == "huggingface_json",
                         "Unsupported tokenizer asset format");
    spec.max_length = value.at("max_length").integer();
    spec.pad_id = static_cast<int32_t>(value.at("pad_id").integer());
    spec.prefix_ids = i32_array(value.find("prefix_ids"));
    spec.suffix_ids = i32_array(value.find("suffix_ids"));
    if (const auto* empty = value.find("empty_input")) {
        vrhino::require(empty->string() == "all_padding", "Unsupported empty-input policy");
        spec.empty_input = vrhino::EmptyInputPolicy::AllPadding;
    }
    if (const auto* suppress = value.find("suppress_metaspace_after_added_token"))
        spec.suppress_metaspace_after_added_token = suppress->boolean();
    if (const auto* added = value.find("added_tokens")) {
        for (const auto& item : added->array()) {
            spec.added_tokens.push_back({item.at("content").string(),
                static_cast<int32_t>(item.at("id").integer()),
                item.at("lstrip").boolean(), item.at("rstrip").boolean()});
        }
    }
    return spec;
}

vrhino::Tensor ids_tensor(const vrhino::TokenizedInput& input) {
    std::vector<int64_t> values(input.input_ids.begin(), input.input_ids.end());
    return vrhino::host_i64({1, static_cast<int64_t>(values.size())}, values);
}

vrhino::Tensor mask_tensor(const vrhino::TokenizedInput& input) {
    return vrhino::host_bool({1, static_cast<int64_t>(input.attention_mask.size())},
                             input.attention_mask);
}

vrhino::Tensor host_mask_slice(const vrhino::Tensor& input, int64_t start, int64_t stop) {
    vrhino::require(input.device().is_host() && input.dtype() == vrhino::DType::Bool &&
                    input.ndim() == 2 && start >= 0 && stop >= start && stop <= input.dim(1),
                    "Invalid host mask slice");
    const int64_t batch = input.dim(0), width = stop - start;
    std::vector<uint8_t> values(static_cast<size_t>(batch * width));
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t column = 0; column < width; ++column)
            values[static_cast<size_t>(row * width + column)] =
                input.data_as<uint8_t>()[row * input.dim(1) + start + column];
    return vrhino::host_bool({batch, width}, values);
}

vrhino::Tensor binding_value(const vrhino::Json& binding,
                             const vrhino::ConditioningComponentResult& result,
                             const vrhino::Tensor& ids, const vrhino::Tensor& mask) {
    const std::string source = binding.at("source").string();
    if (source == "hidden_states") return result.hidden_states;
    if (source == "pooled_output") return result.pooled_output;
    if (source == "input_ids") return ids;
    if (source == "attention_mask") {
        const int64_t start = binding.find("slice_start")
            ? binding.at("slice_start").integer() : 0;
        const int64_t stop = binding.find("slice_stop")
            ? binding.at("slice_stop").integer() : mask.dim(1);
        return host_mask_slice(mask, start, stop);
    }
    throw vrhino::Error("Unsupported conditioning binding source: " + source);
}

void add_static_inputs(vrhino::TensorBundle& input, const vrhino::InputRequest& request,
                       const vrhino::Json& profile) {
    input.emplace("seed", vrhino::scalar_i64(static_cast<int64_t>(request.seed)));
    if (const auto* tensors = profile.find("static_inputs")) {
        for (const auto& [name, value] : tensors->object()) {
            const std::string dtype = value.at("dtype").string();
            std::vector<int64_t> shape;
            for (const auto& item : value.at("shape").array()) shape.push_back(item.integer());
            if (dtype == "i64") {
                std::vector<int64_t> values;
                for (const auto& item : value.at("values").array()) values.push_back(item.integer());
                input.emplace(name, vrhino::host_i64(shape, values));
            } else if (dtype == "f32") {
                std::vector<float> values;
                for (const auto& item : value.at("values").array())
                    values.push_back(static_cast<float>(item.number()));
                input.emplace(name, vrhino::host_f32(shape, values));
            } else throw vrhino::Error("Unsupported static input dtype: " + dtype);
        }
    }
}

std::string source_text(const vrhino::InputRequest& request,
                        const vrhino::Json& component) {
    const std::string source = component.at("text_source").string();
    std::string text;
    if (source == "prompt") text = request.prompt;
    else if (source == "negative_prompt") {
        vrhino::require(request.negative_prompt.has_value(), "Missing required negative prompt");
        text = *request.negative_prompt;
    } else throw vrhino::Error("Unsupported prompt text source: " + source);
    if (const auto* trim = component.find("trim")) {
        if (trim->boolean()) {
            const size_t begin = text.find_first_not_of(" \t\r\n");
            const size_t end = text.find_last_not_of(" \t\r\n");
            text = begin == std::string::npos ? "" : text.substr(begin, end - begin + 1);
        }
    }
    const std::string prefix = component.find("prefix") ? component.at("prefix").string() : "";
    const std::string suffix = component.find("suffix") ? component.at("suffix").string() : "";
    return prefix + text + suffix;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto process_started = std::chrono::steady_clock::now();
        vrhino::require(argc == 3 || argc == 4,
            "usage: vrhino-phase13d-prompt-video PROFILE.json OUTPUT.bundle "
            "[EXTERNAL_INITIAL_STATE.bundle]");
        const vrhino::Json profile = vrhino::Json::parse(read_text(argv[1]));
        vrhino::InputRequest request;
        request.prompt = profile.at("prompt").string();
        if (const auto* negative = profile.find("negative_prompt"))
            request.negative_prompt = negative->string();
        request.seed = static_cast<uint64_t>(profile.at("seed").integer());

        const auto model_started = std::chrono::steady_clock::now();
        vrhino::VrmModel model(profile.at("vrm").string());
        const double model_load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - model_started).count();
        NativeBackend backend;
        vrhino::DType execution_dtype = vrhino::DType::F32;
        if (const auto* dtype = profile.find("execution_dtype")) {
            if (dtype->string() == "bfloat16") execution_dtype = vrhino::DType::BF16;
            else vrhino::require(dtype->string() == "float32",
                                 "execution_dtype must be float32 or bfloat16");
        }
#if VRHINO_USE_METAL_BACKEND
        vrhino::require(execution_dtype == vrhino::DType::F32,
                        "This validation target supports Metal FP32 only");
#endif
        backend.set_execution_dtype(execution_dtype);
        size_t device_budget_gib = 21;
        if (const char* budget_text = std::getenv("VRHINO_DEVICE_BUDGET_GIB"))
            device_budget_gib = static_cast<size_t>(std::strtoull(budget_text, nullptr, 10));
        vrhino::require(device_budget_gib >= 4 && device_budget_gib <= 1024,
                        "VRHINO_DEVICE_BUDGET_GIB is outside [4, 1024]");
        vrhino::MemoryBudget budget{device_budget_gib << 30, 2ULL << 30, 64ULL << 30,
                                    2ULL << 30, 1ULL << 30};
        vrhino::MemoryRuntimeOptions options;
        options.enabled = true;
        options.host_staging = false;
        options.prefetch = false;
        backend.configure_memory_runtime(budget, options);
        const bool profiling = std::getenv("VRHINO_PROFILE") != nullptr;
        backend.enable_profiling(profiling);
        std::map<std::string, vrhino::ProfileStat> profile_rows;
        auto collect_profiles = [&](const std::string& prefix) {
            if (!profiling) return;
            for (auto& [name, stat] : backend.profile_stats())
                profile_rows.emplace(prefix + name, std::move(stat));
        };

        std::vector<vrhino::phase13::SafeTensorSet> assets;
        assets.reserve(profile.at("components").array().size());
        std::map<std::string, size_t> asset_by_path;
        std::vector<size_t> component_asset_indices;
        size_t mapped_bytes = model.file_size();
        const auto asset_mapping_started = std::chrono::steady_clock::now();
        for (const auto& component : profile.at("components").array()) {
            const std::string path = component.at("weights").string();
            auto found = asset_by_path.find(path);
            if (found == asset_by_path.end()) {
                const size_t asset_index = assets.size();
                assets.push_back(path.ends_with(".index.json")
                    ? vrhino::phase13::SafeTensorSet::indexed(path)
                    : vrhino::phase13::SafeTensorSet::single(path));
                mapped_bytes += assets.back().mapped_bytes();
                found = asset_by_path.emplace(path, asset_index).first;
            }
            component_asset_indices.push_back(found->second);
        }
        const double asset_mapping_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - asset_mapping_started).count();
        backend.set_vrm_mapped_bytes(mapped_bytes);

        vrhino::TensorBundle input;
        add_static_inputs(input, request, profile);
        vrhino::TensorBundle observations;
        std::map<std::string, double> tokenizer_seconds;
        std::map<std::string, double> conditioning_seconds;
        size_t index = 0;
        for (const auto& component : profile.at("components").array()) {
            const std::string name = component.at("name").string();
            const auto tokenizer_started = std::chrono::steady_clock::now();
            vrhino::NativeTokenizer tokenizer(
                tokenizer_spec(component.at("tokenizer_spec")),
                read_text(component.at("tokenizer").string()));
            const vrhino::TokenizedInput tokenized =
                tokenizer.encode(source_text(request, component));
            tokenizer_seconds[name] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tokenizer_started).count();
            const vrhino::Tensor ids = ids_tensor(tokenized);
            const vrhino::Tensor mask = mask_tensor(tokenized);
            const auto conditioning_started = std::chrono::steady_clock::now();
            vrhino::ConditioningComponentExecutor executor(
                backend, assets[component_asset_indices[index++]].weights());
            const auto result = executor.execute(
                vrhino::Json::parse(read_text(component.at("graph").string())), ids, mask);
            backend.synchronize();
            conditioning_seconds[name] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - conditioning_started).count();
            collect_profiles("conditioning." + name + ".");
            observations.emplace("tokenizer." + name + ".input_ids", ids);
            observations.emplace("tokenizer." + name + ".attention_mask", mask);
            for (const auto& binding : component.at("bindings").array()) {
                const std::string target = binding.at("target").string();
                vrhino::Tensor value = binding_value(binding, result, ids, mask);
                vrhino::require(value.defined(), "Undefined conditioning binding: " + target);
                vrhino::require(input.emplace(target, value).second,
                                "Duplicate conditioning target: " + target);
                observations.emplace("conditioning." + target,
                    value.device().is_host() ? value : backend.copy_to_host(value));
            }
        }

        auto architecture = vrhino::create_architecture(model);
        const vrhino::PrecisionPolicy precision_policy = profile.find("precision_policy")
            ? vrhino::PrecisionPolicy::from_json(vrhino::Json::parse(
                read_text(profile.at("precision_policy").string())))
            : vrhino::PrecisionPolicy::unqualified_default(execution_dtype);
        vrhino::require(precision_policy.requested_dtype() == execution_dtype,
                        "Precision policy requested dtype does not match execution_dtype");
        vrhino::ComponentExecutionConfig component_execution;
        if (const auto* descriptor = profile.find("component_execution"))
            component_execution =
                vrhino::component_execution_config_from_json(*descriptor);
        vrhino::NativeRuntime runtime(
            backend, precision_policy, std::move(component_execution));
        vrhino::RuntimeResult execution;
        if (argc == 4) {
            const vrhino::TensorBundle external = vrhino::read_bundle(argv[3]);
            vrhino::require(external.size() == 1 && external.contains("initial_latent"),
                            "External initial-state bundle must contain only initial_latent");
            const vrhino::Tensor& initial = external.at("initial_latent");
            vrhino::require(initial.device().is_host(),
                            "External initial state must be a host tensor");
            execution = runtime.execute_with_external_initial_state_for_test(
                *architecture, input, initial);
            execution.outputs.emplace("vae.input", execution.outputs.at("final_latent"));
        } else {
            execution = runtime.execute(*architecture, input);
        }
        collect_profiles("runtime.");
        for (auto& [name, tensor] : observations)
            execution.outputs.emplace(std::move(name), std::move(tensor));
        execution.outputs.emplace("metric.conditioning_mapped_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(mapped_bytes - model.file_size())));
        const auto memory = backend.memory_runtime_stats();
        execution.outputs.emplace("metric.memory.vrm_mapped_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(memory.accounting.vrm_mapped_bytes)));
        execution.outputs.emplace("metric.memory.device_resident_weight_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(memory.accounting.device_resident_weight_bytes)));
        execution.outputs.emplace("metric.memory.peak_device_resident_weight_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(memory.accounting.peak_device_resident_weight_bytes)));
        execution.outputs.emplace("metric.memory.peak_device_activation_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(memory.accounting.peak_device_activation_bytes)));
        execution.outputs.emplace("metric.memory.peak_host_staging_bytes",
            vrhino::scalar_i64(static_cast<int64_t>(memory.accounting.peak_host_staging_bytes)));
        execution.outputs.emplace("metric.memory.prefetch_requests",
            vrhino::scalar_i64(static_cast<int64_t>(memory.prefetch_requests)));
        execution.outputs.emplace("metric.memory.prefetch_hits",
            vrhino::scalar_i64(static_cast<int64_t>(memory.prefetch_hits)));
        execution.outputs.emplace("metric.memory.evictions",
            vrhino::scalar_i64(static_cast<int64_t>(memory.evictions)));
        execution.outputs.emplace("metric.memory.forced_syncs",
            vrhino::scalar_i64(static_cast<int64_t>(memory.forced_syncs)));
        execution.outputs.emplace("metric.component_execution.mode",
            vrhino::scalar_i64(static_cast<int64_t>(
                execution.component_execution.mode)));
        execution.outputs.emplace("metric.component_execution.temporal_tiles",
            vrhino::scalar_i64(static_cast<int64_t>(
                execution.component_execution.tiling.temporal_tiles)));
        execution.outputs.emplace(
            "metric.component_execution.spatial_tiles_per_temporal",
            vrhino::scalar_i64(static_cast<int64_t>(execution.component_execution
                .tiling.spatial_tiles_per_temporal)));
        execution.outputs.emplace("metric.component_execution.graph_executions",
            vrhino::scalar_i64(static_cast<int64_t>(
                execution.component_execution.tiling.graph_executions)));
        const auto bundle_started = std::chrono::steady_clock::now();
        vrhino::write_bundle(argv[2], execution.outputs);
        const double bundle_write_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - bundle_started).count();
        const double process_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - process_started).count();
        std::cout << "status=PASS\nprofile=" << profile.at("id").string()
                  << "\nprecomputed_conditioning_inputs=0\npython_fallback=0"
                  << "\nmodel_specific_prompt_runtime_loc=0"
                  << "\nmodel_specific_conditioning_runtime_loc=0"
                  << "\nbackend=" << kBackendName
                  << "\nexecution_dtype=" << vrhino::dtype_name(execution_dtype)
                  << "\nprecision_policy="
                  << (profile.find("precision_policy")
                      ? profile.at("precision_policy").string() : "unqualified_default")
                  << "\narchitecture_specific_backend_kernels=0"
                  << "\nexternal_initial_state=" << (argc == 4 ? 1 : 0)
                  << "\nweight_cache_hits=" << backend.weight_cache_hits()
                  << "\nweight_cache_misses=" << backend.weight_cache_misses()
                  << "\nvrm_mapped_bytes=" << memory.accounting.vrm_mapped_bytes
                  << "\ndevice_resident_weight_bytes="
                  << memory.accounting.device_resident_weight_bytes
                  << "\npeak_device_bytes=" << memory.accounting.peak_device_bytes
                  << "\npeak_device_weight_bytes="
                  << memory.accounting.peak_device_resident_weight_bytes
                  << "\npeak_device_activation_bytes="
                  << memory.accounting.peak_device_activation_bytes
                  << "\npeak_host_staging_bytes="
                  << memory.accounting.peak_host_staging_bytes
                  << "\nmemory_prefetch_requests=" << memory.prefetch_requests
                  << "\nmemory_prefetch_hits=" << memory.prefetch_hits
                  << "\nmemory_evictions=" << memory.evictions
                  << "\nmemory_forced_syncs=" << memory.forced_syncs
                  << "\nprofiler_enabled=" << (profiling ? "true" : "false")
                  << "\nmodel_load_seconds=" << model_load_seconds
                  << "\nasset_mapping_seconds=" << asset_mapping_seconds
                  << "\nbundle_write_seconds=" << bundle_write_seconds
                  << "\nprocess_seconds=" << process_seconds
                  << "\nsampling_seconds=" << execution.sampling_seconds
                  << "\ndecode_seconds=" << execution.decode_seconds
                  << "\nexecution_seconds=" << execution.execution_seconds
                  << "\nscheduler_seconds=" << execution.scheduler_seconds
                  << "\nmemory_upload_copies=" << memory.upload_copies
                  << "\nmemory_upload_bytes=" << memory.upload_bytes
                  << "\nmemory_upload_seconds=" << memory.upload_seconds
                  << "\nmemory_upload_bandwidth_gbps=" << memory.upload_bandwidth_gbps()
                  << "\nmemory_overlap_ratio=" << memory.overlap_ratio()
                  << "\nmemory_stream_waits=" << memory.stream_waits
                  << "\nmemory_event_waits=" << memory.event_waits << '\n';
        for (const auto& [name, seconds] : tokenizer_seconds)
            std::cout << "tokenizer_seconds=" << name << ',' << seconds << '\n';
        for (const auto& [name, seconds] : conditioning_seconds)
            std::cout << "conditioning_seconds=" << name << ',' << seconds << '\n';
        for (size_t step = 0; step < execution.denoiser_call_seconds.size(); ++step)
            std::cout << "denoiser_step_seconds=" << step << ','
                      << execution.denoiser_call_seconds[step] << '\n';
        for (const auto& [name, stat] : profile_rows)
            std::cout << "profile_stat=" << name
                      << ",calls=" << stat.calls
                      << ",cuda_ms=" << stat.device_milliseconds
                      << ",mean_ms=" << stat.mean_milliseconds
                      << ",p50_ms=" << stat.p50_milliseconds
                      << ",p95_ms=" << stat.p95_milliseconds
                      << ",min_ms=" << stat.minimum_milliseconds
                      << ",max_ms=" << stat.maximum_milliseconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vrhino-phase13d-prompt-video: " << error.what() << '\n';
        return 1;
    }
}
