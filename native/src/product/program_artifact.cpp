#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/tensor_util.h"
#include <bit>
#include <fstream>

namespace vrhino::product {
VerifiedDeclaredPrograms admit_verified_program_artifact(const std::filesystem::path &path,
                                                         uint64_t bytes, const std::string &sha,
                                                         const VrmModel &model) {
    require(bytes > 0 && bytes <= 1024 * 1024 && std::filesystem::file_size(path) == bytes,
            "Invalid program artifact size");
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "Missing programs artifact");
    std::string contents(bytes + 1, '\0');
    f.read(contents.data(), contents.size());
    require(f.eof() && f.gcount() == static_cast<std::streamsize>(bytes),
            "Programs artifact size changed");
    contents.resize(bytes);
    require(sha256_bytes(contents) == sha, "Programs artifact SHA256 mismatch");
    JsonParseLimits limits;
    limits.maximum_document_bytes = 1024 * 1024;
    limits.maximum_container_entries = 10001;
    limits.maximum_total_values = 100000;
    const auto j = Json::parse(contents, limits);
    require(j.serialize() == model.metadata().at("programs").serialize(),
            "Program artifact/embedded authority mismatch");
    const auto package = PackageDeclaration::parse(model.graph());
    auto programs = admit_program_declaration(j, package);
    Json::Array steps;
    for (int i = 0; i < programs.sampling.steps; ++i) {
        const auto &t = programs.sampling.model_timestep_at(i);
        Json time;
        if (t.dtype() == DType::F32)
            time = Json(
                Json::Object{{"dtype", Json(std::string("float32"))},
                             {"shape", Json(Json::Array{Json(int64_t(1)), Json(int64_t(1))})},
                             {"value", Json(double(read_scalar_f32(t)))},
                             {"bits", Json(int64_t(std::bit_cast<uint32_t>(read_scalar_f32(t))))}});
        else
            time = Json(read_scalar_i64(t));
        const auto &transition = programs.sampling.contract->schedule.flow_at(i);
        const auto id = j.at("execution").at("instance_ids").array().at(i).integer();
        steps.emplace_back(Json::Object{
            {"step", Json(int64_t(i))},
            {"instance_id", Json(id)},
            {"binding_id",
             Json(int64_t(package.instances().at(static_cast<uint32_t>(id)).binding))},
            {"model_timestep", time},
            {"sigma", Json(double(transition.sigma))},
            {"next_sigma", Json(double(transition.next_sigma))},
            {"guidance", Json(double(programs.sampling.guidance_schedule->at(i).scale))}});
    }
    return {std::move(programs), Json(Json::Object{{"programs_sha256", Json(sha)},
                                                   {"declaration_schema", j.at("schema")},
                                                   {"sampling", j.at("sampling")},
                                                   {"steps", Json(steps)},
                                                   {"numerically_qualified", Json(false)}})};
}
} // namespace vrhino::product
