#include "builtin_pb/sentencepiece_model.pb.h"
#include "ltx_schema2_test_support.h"
#include "vrhino/product/component_preparation.h"
#include "vrhino/product/prepared_execution.h"
#include "vrhino/product/token_grid_geometry.h"
#include "vrhino/tokenizer.h"
#include <fstream>
#include <iostream>
namespace s = vrhino::ltx_schema2_test;
namespace p = vrhino::product;
using namespace vrhino;
void shard(const s::fs::path &path, const std::string &name) {
    std::string h = "{\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    while (h.size() % 8) {
        h += ' ';
    }
    std::ofstream f(path, std::ios::binary);
    uint64_t n = h.size();
    f.write(reinterpret_cast<const char *>(&n), 8);
    f << h;
    float x[2] = {1, 2};
    f.write(reinterpret_cast<char *>(x), 8);
}
// Generated offline, tiny identity-normalized vocabulary. The optional external
// fixture exercises real tokenizer bytes without putting model assets in Git.
std::string synthetic_sentencepiece() {
    sentencepiece::ModelProto model;
    auto *trainer = model.mutable_trainer_spec();
    trainer->set_model_type(sentencepiece::TrainerSpec::UNIGRAM);
    trainer->set_pad_id(0);
    trainer->set_eos_id(1);
    trainer->set_unk_id(2);
    trainer->set_bos_id(-1);
    const auto add = [&](const std::string &text,
                         sentencepiece::ModelProto::SentencePiece::Type type) {
        auto *piece = model.add_pieces();
        piece->set_piece(text);
        piece->set_score(-1);
        piece->set_type(type);
    };
    add("<pad>", sentencepiece::ModelProto::SentencePiece::CONTROL);
    add("</s>", sentencepiece::ModelProto::SentencePiece::CONTROL);
    add("<unk>", sentencepiece::ModelProto::SentencePiece::UNKNOWN);
    add("▁", sentencepiece::ModelProto::SentencePiece::NORMAL);
    for (char c = 'a'; c <= 'z'; ++c)
        add(std::string(1, c), sentencepiece::ModelProto::SentencePiece::NORMAL);
    trainer->set_vocab_size(model.pieces_size());
    model.mutable_normalizer_spec()->set_name("identity");
    std::string serialized;
    require(model.SerializeToString(&serialized), "Synthetic SentencePiece serialization failed");
    return serialized;
}
struct AdmissionOnly final : Architecture {
    std::unique_ptr<Denoiser> create_denoiser(Backend &, const PrecisionPolicy &,
                                              const TensorBundle &) override {
        throw Error("Unexpected endpoint");
    }
    SamplingProgram create_program(const TensorBundle &) override { return {}; }
    Tensor decode(Backend &, const PrecisionPolicy &, const Tensor &,
                  const TensorBundle &) override {
        throw Error("Unexpected decoder");
    }
};
int main(int argc, char **argv) {
    try {
        require(argc == 2 || argc == 3,
                "usage: ltx-schema2-integration-tests OUTPUT [SENTENCEPIECE]");
        s::fs::path d = argv[1];
        s::fs::create_directories(d);
        shard(d / "one", "one");
        shard(d / "two", "two");
        auto index = Json::parse(R"({"weight_map":{"one":"a.safetensors","two":"b.safetensors"}})");
        std::map<std::string, p::VerifiedSafeTensorArtifact> artifacts{
            {"a.safetensors", {d / "one", s::fs::file_size(d / "one"), p::sha256_file(d / "one")}},
            {"b.safetensors", {d / "two", s::fs::file_size(d / "two"), p::sha256_file(d / "two")}}};
        auto admitted = p::SafeTensorAsset::indexed_verified(index, artifacts);
        admitted.verify_backing_identity();
        require(admitted.weights().at("one").data_as<float>()[0] == 1, "Indexed binding mismatch");
        s::reject("missing shard", [&] {
            auto a = artifacts;
            a.erase(a.begin());
            p::SafeTensorAsset::indexed_verified(index, a);
        });
        s::reject("unused shard", [&] {
            auto a = artifacts;
            a.emplace("extra.safetensors", a.begin()->second);
            p::SafeTensorAsset::indexed_verified(index, a);
        });
        s::reject("wrong shard size", [&] {
            auto a = artifacts;
            a.begin()->second.bytes++;
            p::SafeTensorAsset::indexed_verified(index, a);
        });
        s::reject("wrong shard SHA", [&] {
            auto a = artifacts;
            a.begin()->second.sha256 = std::string(64, '0');
            p::SafeTensorAsset::indexed_verified(index, a);
        });
        s::reject("tensor missing from index", [&] {
            p::SafeTensorAsset::indexed_verified(
                Json::parse(R"({"weight_map":{"absent":"a.safetensors","two":"b.safetensors"}})"),
                artifacts);
        });
        s::reject("tensor assigned wrong shard", [&] {
            p::SafeTensorAsset::indexed_verified(
                Json::parse(R"({"weight_map":{"one":"b.safetensors","two":"a.safetensors"}})"),
                artifacts);
        });
        {
            std::fstream f(d / "one", std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(-1, std::ios::end);
            f.put('x');
        }
        // Both asset and mapping moves must preserve strict validation state.
        auto moved = std::move(admitted);
        s::reject("backing drift after admission and move",
                  [&] { moved.verify_backing_identity(); });
        std::string blob = synthetic_sentencepiece();
        if (argc == 3) {
            std::ifstream file(argv[2], std::ios::binary);
            require(file.good(), "Missing SP fixture");
            blob.assign(std::istreambuf_iterator<char>(file), {});
        }
        require(!blob.empty(), "Empty SP fixture");
        TokenizerSpec spec;
        spec.format = TokenizerAssetFormat::SentencePiece;
        spec.max_length = 128;
        spec.pad_id = 0;
        spec.suffix_ids = {1};
        NativeTokenizer tokenizer(spec, blob);
        require(tokenizer.token_matches(0, "<pad>") && tokenizer.token_matches(1, "</s>"),
                "SP token identities");
        require(!tokenizer.token_matches(1, "<pad>"), "SP false identity");
        s::reject("corrupt SentencePiece", [&] { NativeTokenizer invalid(spec, "not a model"); });
        s::reject("invalid UTF8", [&] { tokenizer.encode(std::string("\xff")); });
        for (const auto &text :
             {std::string(""), std::string("A red cube on a table."), std::string(2048, 'a')}) {
            const auto tokens = tokenizer.encode(text);
            require(tokens.input_ids.size() == 128 && tokens.attention_mask.size() == 128 &&
                        tokens.valid_length > 0 && tokens.valid_length <= 128,
                    "Padded token contract");
            require(tokens.input_ids.at(tokens.valid_length - 1) == 1, "EOS contract");
            for (int i = 0; i < 128; ++i)
                require(tokens.attention_mask[i] == (i < tokens.valid_length),
                        "Bool validity contract");
        }
        auto geometry =
            p::admit_token_grid_geometry(s::geometry(), 704, 480, 121, 5703, 2147483648);
        require(geometry.latent_shape == std::vector<int64_t>({1, 5280, 128}),
                "Production geometry");
        p::PreparedTextProductRequest request;
        request.runtime_inputs = geometry.runtime_inputs;
        request.conditioning.push_back({"positive",
                                        Json(Json::Object{}),
                                        Tensor::host({1, 128}, DType::I64),
                                        host_bool({1, 128}, std::vector<uint8_t>(128, 1)),
                                        {1, 128, 4096},
                                        "positive_mask"});
        request.conditioning.push_back({"negative",
                                        Json(Json::Object{}),
                                        Tensor::host({1, 128}, DType::I64),
                                        host_bool({1, 128}, std::vector<uint8_t>(128, 1)),
                                        {1, 128, 4096},
                                        "negative_mask"});
        auto hidden = Tensor::host({1, 128, 4096}, DType::F32);
        std::memset(hidden.data(), 0, hidden.bytes());
        auto bound = request.bind_conditioning_outputs({hidden, hidden});
        require(bound.size() == 9 && bound.at("positive_mask").dtype() == DType::Bool,
                "Hidden/mask binding contract");
        s::reject("trimmed hidden", [&] {
            request.bind_conditioning_outputs({Tensor::host({1, 127, 4096}, DType::F32), hidden});
        });
        s::reject("wrong hidden dtype", [&] {
            request.bind_conditioning_outputs({Tensor::host({1, 128, 4096}, DType::I64), hidden});
        });
        request.conditioning[0].attention_mask = Tensor::host({1, 128}, DType::I64);
        s::reject("wrong mask dtype", [&] { request.bind_conditioning_outputs({hidden, hidden}); });
        request.conditioning[0].attention_mask = host_bool({1, 128}, std::vector<uint8_t>(128, 0));
        s::reject("all invalid mask", [&] { request.bind_conditioning_outputs({hidden, hidden}); });
        request.conditioning[0].attention_mask.data_as<uint8_t>()[0] = 2;
        s::reject("nonboolean mask byte",
                  [&] { request.bind_conditioning_outputs({hidden, hidden}); });
        request.owner = std::make_shared<p::AdmittedLocalProduct>();
        request.owner->architecture = std::make_unique<AdmissionOnly>();
        int factories = 0;
        s::reject("missing conditioning owner before GPU", [&] {
            p::execute_prepared_product(request, PrecisionPolicy::fp32(),
                                        [&]() -> std::unique_ptr<Backend> {
                                            ++factories;
                                            throw Error("Unexpected factory");
                                        });
        });
        require(factories == 0, "Missing owner allocated execution resources");
        request.owner->conditioning = std::make_unique<p::SafeTensorAsset>(std::move(moved));
        request.sampling.steps = 2;
        s::reject("changed backing before GPU", [&] {
            p::execute_prepared_product(request, PrecisionPolicy::fp32(),
                                        [&]() -> std::unique_ptr<Backend> {
                                            ++factories;
                                            throw Error("Unexpected factory");
                                        });
        });
        require(factories == 0, "Changed backing allocated execution resources");
        std::cout << "LTX_SCHEMA2_INTEGRATION=PASS negative_cases=" << s::negatives
                  << " native_sentencepiece=PASS strict_shards=PASS ownership=PASS "
                     "padded_mask=PASS failed_admission_before_gpu=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
