#include "vrhino/product/qualification_precision.h"
#include "vrhino/error.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace vrhino;
namespace p = vrhino::product;
int main(int argc, char** argv) { try {
    require(argc == 3, "usage: qualification-precision-tests FIXTURE OUTPUT");
    const std::filesystem::path dir = argv[2];
    std::filesystem::create_directories(dir);
    const auto path = dir / "policy.json";
    std::ifstream f(argv[1], std::ios::binary);
    require(f.good(), "Cannot read fixture");
    const std::string bytes(std::istreambuf_iterator<char>(f), {});
    const auto write = [&](const std::string& text) { std::ofstream out(path, std::ios::binary); out << text; };
    write(bytes);
    require(p::sha256_bytes("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "Empty digest vector");
    require(p::sha256_bytes("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "Digest vector");
    require(p::sha256_bytes(bytes) == p::sha256_file(path), "Hash bytes/file consistency");
    p::ResolvedRunnableModel model;
    model.artifacts.emplace("policy", p::ResolvedArtifact{
        {"policy", "precision.policy", "policy.json", bytes.size(), p::sha256_bytes(bytes), true}, path});
    const auto loaded = p::load_qualification_precision(model, "policy", "bf16");
    require(loaded.policy.requested_dtype() == DType::BF16, "Requested mode lost");
    require(loaded.policy.persistent_state_dtype(PrecisionSemantic::SamplingState) == DType::BF16,
            "Sampling role changed");
    require(loaded.policy.persistent_state_dtype(PrecisionSemantic::ResidualState) == DType::F32,
            "Residual role changed");
    require(loaded.policy.has_scalar_role(PrecisionScalarRole::SolverCoefficient), "Declared scalar role lost");
    require(loaded.evidence.at("declaration").serialize() == Json::parse(bytes).serialize(), "Policy evidence lost");
    require(!loaded.evidence.at("numerical_admission_granted").boolean(), "Qualification promoted ordinary run");
    int rejected = 0;
    const auto reject = [&](auto fn, const std::string& message) {
        bool caught = false;
        try { fn(); } catch (const std::exception& e) { caught = std::string(e.what()).find(message) != std::string::npos; }
        require(caught, "Expected failure: " + message); ++rejected;
    };
    const auto load = [&] { return p::load_qualification_precision(model, "policy", "bf16"); };
    reject([&] { p::load_qualification_precision(model, "missing", "bf16"); }, "Missing qualification precision artifact");
    reject([&] { p::load_qualification_precision(model, "", "bf16"); }, "artifact ID");
    reject([&] { p::load_qualification_precision(model, "policy", "fp16"); }, "BF16 mode");
    const auto valid = model;
    model.artifacts.at("policy").declaration.required = false;
    reject(load, "artifact declaration"); model = valid;
    model.artifacts.at("policy").declaration.role = "runtime";
    reject(load, "artifact declaration"); model = valid;
    model.artifacts.at("policy").declaration.id = "other";
    reject(load, "artifact declaration"); model = valid;
    for (auto size : {uint64_t(0), uint64_t(65537)}) {
        model.artifacts.at("policy").declaration.size = size;
        reject(load, "artifact size"); model = valid;
    }
    auto corrupt = bytes; corrupt[0] = ' '; write(corrupt);
    reject(load, "SHA256 mismatch"); write(bytes + " ");
    reject(load, "size mismatch"); write(bytes.substr(1));
    reject(load, "size mismatch"); write(bytes);
    std::filesystem::remove(path); reject(load, "Cannot read"); write(bytes);
    const auto invalid_document = [&](std::string text, const std::string& message) {
        write(text); model.artifacts.at("policy").declaration.size = text.size();
        model.artifacts.at("policy").declaration.sha256 = p::sha256_bytes(text);
        reject(load, message); model = valid; write(bytes);
    };
    auto fields = Json::parse(bytes).object();
    fields["schema"] = Json(std::string("unknown"));
    invalid_document(Json(fields).serialize(), "schema mismatch");
    fields = Json::parse(bytes).object(); fields["requested_mode"] = Json(std::string("fp32"));
    invalid_document(Json(fields).serialize(), "mode mismatch");
    auto invalid_role = bytes;
    const auto position = invalid_role.find("SOLVER_COEFFICIENT");
    require(position != std::string::npos, "Fixture scalar role absent");
    invalid_role.replace(position, std::string("SOLVER_COEFFICIENT").size(), "UNKNOWN_COEFFICIENT");
    invalid_document(invalid_role, "Unknown precision scalar role");
#ifndef _WIN32
    for (const auto* name : {"VRHINO_BF16_LINEAR_OUTPUT", "VRHINO_CUDA_ATTENTION_VARIANT", "VRHINO_ATTENTION_CAPTURE_EXIT"}) {
        require(std::getenv(name) == nullptr, "Test requires clean environment");
        setenv(name, "", 1); reject(load, "Research switch"); unsetenv(name);
    }
#endif
    std::cout << "PASS declared BF16 roles, immutable byte identity, legacy-independent admission; " << rejected << " negative cases\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
