#include <array>
#include <barrier>
#include <future>
#include <sys/stat.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "vrhino/product/converter.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/vrm_verification.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void put_u64(std::ostream& output, const uint64_t value) {
    std::array<char, 8> bytes{};
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<char>((value >> (8 * index)) & 0xff);
    output.write(bytes.data(), bytes.size());
}

void write_safetensors(const fs::path& path, const uint16_t value) {
    std::string header =
        "{\"x\":{\"data_offsets\":[0,2],\"dtype\":\"BF16\",\"shape\":[1]}}";
    while (header.size() % 8 != 0) header.push_back(' ');
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    put_u64(output, header.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<char, 2> payload{
        static_cast<char>(value & 0xff), static_cast<char>(value >> 8)};
    output.write(payload.data(), payload.size());
    require_test(static_cast<bool>(output), "cannot write safetensors fixture");
}

vrhino::Json object(
        std::initializer_list<std::pair<const std::string, vrhino::Json>> values) {
    vrhino::Json::Object result;
    for (const auto& [key, value] : values) result.emplace(key, value);
    return vrhino::Json(vrhino::Json::Value(std::move(result)));
}

fs::path write_vrm(const fs::path& root, const std::string& name,
                   const std::string& architecture, const uint16_t value) {
    const fs::path source = root / (name + ".safetensors");
    const fs::path output = root / (name + ".vrm");
    write_safetensors(source, value);
    product::SafeTensorReader reader(source);
    const product::TensorMapping mapping{
        "x", vrhino::DType::BF16, {1}, "identity_bytes", "component.x",
        vrhino::DType::BF16, {1}, "component", "weight"};
    const vrhino::Json metadata = object({
        {"architecture", vrhino::Json(vrhino::Json::Value(architecture))},
    });
    const vrhino::Json graph = object({
        {"schema_version", vrhino::Json(vrhino::Json::Value(int64_t{1}))},
    });
    (void)product::write_vrm_streaming(
        output, "component", architecture, metadata, graph, reader, {mapping});
    return output;
}

product::VrmComponentIntegrityContract contract_for(
        const fs::path& path, const std::string& architecture) {
    return {path.stem().string(), architecture, fs::file_size(path),
            product::sha256_file(path)};
}

template <typename Operation>
void expect_failure(Operation&& operation, const std::string& context) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(context + ": operation unexpectedly succeeded");
}

void change_byte(const fs::path& path, const std::streamoff offset) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(offset);
    char value = 0;
    file.read(&value, 1);
    require_test(static_cast<bool>(file), "cannot read mutation byte");
    value ^= 0x5a;
    file.seekp(offset);
    file.write(&value, 1);
    file.flush();
    require_test(static_cast<bool>(file), "cannot write mutation byte");
}

struct stat stat_file(const fs::path& path) {
    struct stat value{};
    require_test(stat(path.c_str(), &value) == 0, "cannot stat fixture");
    return value;
}

bool same_time(const timespec& left, const timespec& right) {
    return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

// Synchronize to a filesystem tick before changing the opened file's metadata.
// A sleep alone could leave the old ctime-equality bug passing by chance.
void await_filesystem_tick(const fs::path& root, const timespec& prior) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    const auto sentinel = root / "clock-sentinel";
    do {
        { std::ofstream output(sentinel); output << 'x'; }
        const auto now = stat_file(sentinel).st_ctim;
        if (now.tv_sec > prior.tv_sec ||
            (now.tv_sec == prior.tv_sec && now.tv_nsec > prior.tv_nsec)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("fixture filesystem clock did not advance");
}

void require_payload(const vrhino::VrmModel& model, const uint16_t expected) {
    const auto& tensor = model.tensor("component.x");
    require_test(tensor.bytes() == 2, "wrong fixture tensor length");
    const auto* bytes = static_cast<const unsigned char*>(tensor.data());
    require_test(bytes[0] == (expected & 0xff) && bytes[1] == (expected >> 8),
                 "opened tensor content was silently swapped");
}

void stable_namespace_tests(const fs::path& root) {
    for (const std::string operation : {"rename", "replace-same-size",
            "replace-different-size", "unlink", "recreate", "metadata"}) {
        const auto active = write_vrm(root, operation, "fixture_a", 0x1234);
        const auto other = write_vrm(root, operation + "-other", "fixture_a", 0x5678);
        const auto moved = root / (operation + "-moved");
        const auto contract = contract_for(active, "fixture_a");
        if (operation == "replace-different-size") {
            std::ofstream output(other, std::ios::app | std::ios::binary);
            output << "invalid replacement size";
        } else {
            require_test(fs::file_size(active) == fs::file_size(other),
                         "same-size replacement fixture differs in length");
        }
        const auto before = stat_file(active);
        const int observer = open(active.c_str(), O_RDONLY | O_CLOEXEC);
        require_test(observer >= 0, "cannot open identity observer");
        product::VrmVerificationObservation observation{};
        unsigned hooks = 0;
        std::unique_ptr<vrhino::VrmModel> model;
        try {
            model = product::load_verified_vrm_component(active, contract, &observation, [&] {
                ++hooks;
                await_filesystem_tick(root, before.st_ctim);
                if (operation == "rename") fs::rename(active, moved);
                if (operation == "replace-same-size" || operation == "replace-different-size")
                    fs::rename(other, active);
                if (operation == "unlink" || operation == "recreate") fs::remove(active);
                if (operation == "recreate") fs::copy_file(other, active);
                if (operation == "metadata")
                    require_test(chmod(active.c_str(), 0400) == 0, "fixture chmod failed");
                struct stat after{};
                require_test(fstat(observer, &after) == 0, "observer fstat failed");
                require_test(before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
                             before.st_size == after.st_size &&
                             same_time(before.st_mtim, after.st_mtim) &&
                             !same_time(before.st_ctim, after.st_ctim),
                             "fixture did not isolate a ctime-only identity change");
                require_test(product::sha256_file_descriptor(observer, contract.bytes) ==
                             contract.sha256, "namespace operation changed opened content");
                if (operation == "rename" || operation == "unlink") {
                    require_test(!fs::exists(active), "old pathname survived removal");
                } else if (operation != "metadata") {
                    const auto path_after = stat_file(active);
                    require_test(path_after.st_ino != after.st_ino ||
                                 path_after.st_dev != after.st_dev,
                                 "replacement did not change pathname inode");
                    require_test(product::sha256_file(active) != contract.sha256,
                                 "replacement must contain different bytes");
                }
            });
        } catch (...) { close(observer); throw; }
        close(observer);
        require_payload(*model, 0x1234);
        require_test(hooks == 1, "stable-open hook repeated on retry");
        require_test(observation.sha256_completed && observation.blake2b_completed &&
                     observation.sha256_bytes == 2 * contract.bytes &&
                     observation.blake2b_bytes == 2 * (contract.bytes - 128) &&
                     observation.maximum_digest_consumers == 2,
                     "ctime re-verification work/concurrency accounting");
        if (operation == "replace-same-size" || operation == "replace-different-size" ||
            operation == "recreate") {
            expect_failure([&] {
                (void)product::load_verified_vrm_component(active, contract);
            }, "new open accepted malicious replacement under old contract");
        }
        std::cout << "stable namespace " << operation << ": PASS\n";
    }

    // A self-consistent replacement VRM at the SAME inode still needs the
    // external expected SHA, even if the attacker restores mtime.
    const auto active = write_vrm(root, "restored-mtime", "fixture_a", 0x1234);
    const auto other = write_vrm(root, "restored-mtime-other", "fixture_a", 0x5678);
    const auto contract = contract_for(active, "fixture_a");
    const auto before = stat_file(active);
    product::VrmVerificationObservation observation{};
    bool rejected_sha = false;
    try {
        (void)product::load_verified_vrm_component(active, contract, &observation, [&] {
            await_filesystem_tick(root, before.st_ctim);
            std::ifstream source(other, std::ios::binary);
            std::fstream target(active, std::ios::in | std::ios::out | std::ios::binary);
            target << source.rdbuf();
            target.flush();
            require_test(static_cast<bool>(target), "cannot overwrite same inode");
            const timespec times[2] = {before.st_atim, before.st_mtim};
            require_test(utimensat(AT_FDCWD, active.c_str(), times, 0) == 0,
                         "cannot restore mtime");
            const auto after = stat_file(active);
            require_test(before.st_ino == after.st_ino && before.st_size == after.st_size &&
                         same_time(before.st_mtim, after.st_mtim) &&
                         !same_time(before.st_ctim, after.st_ctim), "mutation fixture identity");
        });
    } catch (const product::ModelPackageError& error) {
        rejected_sha = std::string(error.what()).find("SHA-256 mismatch") != std::string::npos;
    }
    require_test(rejected_sha && observation.sha256_completed && observation.blake2b_completed,
                 "valid in-place replacement with restored mtime escaped external SHA");
    std::cout << "same inode valid replacement with restored mtime: PASS\n";

    const auto shared = write_vrm(root, "readers", "fixture_a", 0x1234);
    const auto replacement = write_vrm(root, "readers-other", "fixture_a", 0x5678);
    const auto shared_contract = contract_for(shared, "fixture_a");
    const auto shared_before = stat_file(shared);
    std::barrier rendezvous(5);
    std::vector<std::future<void>> readers;
    for (int index = 0; index < 4; ++index) {
        readers.push_back(std::async(std::launch::async, [&] {
            const auto loaded = product::load_verified_vrm_component(
                shared, shared_contract, nullptr, [&] {
                    rendezvous.arrive_and_wait();
                    rendezvous.arrive_and_wait();
                });
            require_payload(*loaded, 0x1234);
        }));
    }
    rendezvous.arrive_and_wait(); // All four authoritative descriptors are open.
    await_filesystem_tick(root, shared_before.st_ctim);
    fs::rename(replacement, shared);
    rendezvous.arrive_and_wait();
    for (auto& reader : readers) reader.get();
    std::cout << "four concurrent readers across atomic replacement: PASS\n";
}

}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-vrm-verification-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
    try {
        const fs::path valid = write_vrm(root, "valid", "fixture_a", 0x1234);
        const auto valid_contract = contract_for(valid, "fixture_a");
        product::VrmVerificationObservation success{};
        const auto model = product::load_verified_vrm_component(
            valid, valid_contract, &success);
        require_test(model->architecture_id() == "fixture_a", "valid architecture");
        require_test(success.sha256_completed && success.blake2b_completed,
                     "both valid digest paths did not complete");
        require_test(success.sha256_bytes == fs::file_size(valid) &&
                         success.blake2b_bytes + 128 == fs::file_size(valid),
                     "valid digest byte accounting");
        require_test(success.maximum_digest_consumers == 2,
                     "digest concurrency bound");

        auto bad_sha_contract = valid_contract;
        bad_sha_contract.sha256 = std::string(64, '0');
        product::VrmVerificationObservation bad_sha_observation{};
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                valid, bad_sha_contract, &bad_sha_observation);
        }, "SHA-only mismatch");
        require_test(bad_sha_observation.sha256_completed &&
                         bad_sha_observation.blake2b_completed,
                     "SHA mismatch returned before both paths joined");

        const fs::path bad_blake = root / "bad-blake.vrm";
        fs::copy_file(valid, bad_blake);
        change_byte(bad_blake, static_cast<std::streamoff>(fs::file_size(bad_blake) - 1));
        const auto bad_blake_contract = contract_for(bad_blake, "fixture_a");
        product::VrmVerificationObservation bad_blake_observation{};
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                bad_blake, bad_blake_contract, &bad_blake_observation);
        }, "BLAKE-only mismatch");
        require_test(bad_blake_observation.sha256_completed,
                     "BLAKE mismatch did not join SHA worker");

        const fs::path both_bad = root / "both-bad.vrm";
        fs::copy_file(valid, both_bad);
        const auto both_contract = contract_for(both_bad, "fixture_a");
        change_byte(both_bad, static_cast<std::streamoff>(fs::file_size(both_bad) - 1));
        product::VrmVerificationObservation both_observation{};
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                both_bad, both_contract, &both_observation);
        }, "combined digest mismatch");
        require_test(both_observation.sha256_completed,
                     "combined mismatch did not complete SHA path");

        const fs::path truncated = root / "truncated.vrm";
        fs::copy_file(valid, truncated);
        fs::resize_file(truncated, fs::file_size(truncated) - 1);
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                truncated, contract_for(truncated, "fixture_a"));
        }, "truncated VRM");

        const fs::path malformed = root / "malformed.vrm";
        fs::copy_file(valid, malformed);
        change_byte(malformed, 0);
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                malformed, contract_for(malformed, "fixture_a"));
        }, "malformed VRM");

        expect_failure([&] {
            (void)product::sha256_file_descriptor(-1, 1);
        }, "SHA worker read error");
        const fs::path empty = root / "empty";
        { std::ofstream output(empty, std::ios::binary); }
        const int empty_descriptor = open(empty.c_str(), O_RDONLY | O_CLOEXEC);
        require_test(empty_descriptor >= 0, "cannot open empty fixture");
        expect_failure([&] {
            (void)product::sha256_file_descriptor(empty_descriptor, 1);
        }, "SHA premature EOF");
        close(empty_descriptor);

        const fs::path changed_size = root / "changed-size.vrm";
        fs::copy_file(valid, changed_size);
        const auto changed_size_contract = contract_for(changed_size, "fixture_a");
        product::VrmVerificationObservation changed_size_observation{};
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                changed_size, changed_size_contract, &changed_size_observation, [&] {
                    fs::resize_file(changed_size, fs::file_size(changed_size) - 1);
                });
        }, "concurrent SHA premature EOF and loader error");
        require_test(!changed_size_observation.sha256_completed &&
                         !changed_size_observation.blake2b_completed,
                     "concurrent worker failures were not retained");

        auto invalid_digest_contract = valid_contract;
        invalid_digest_contract.sha256 = "not-a-sha256";
        expect_failure([&] {
            (void)product::load_verified_vrm_component(valid, invalid_digest_contract);
        }, "invalid expected SHA representation");

        const fs::path replacement_a = write_vrm(
            root, "replacement-a", "fixture_a", 0x2345);
        const fs::path replacement_b = write_vrm(
            root, "replacement-b", "fixture_b", 0x3456);
        const auto replacement_a_contract = contract_for(replacement_a, "fixture_a");
        const auto replacement_b_contract = contract_for(replacement_b, "fixture_b");
        const fs::path opened_a = root / "opened-a.vrm";
        const fs::path active = root / "active.vrm";
        fs::rename(replacement_a, active);
        const auto stable_model = product::load_verified_vrm_component(
            active, replacement_a_contract, nullptr, [&] {
                await_filesystem_tick(root, stat_file(active).st_ctim);
                fs::rename(active, opened_a);
                fs::rename(replacement_b, active);
            });
        require_test(stable_model->architecture_id() == "fixture_a",
                     "path replacement escaped the stable open identity");
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                opened_a, replacement_b_contract);
        }, "mixed A/B identity");

        const fs::path disappearing = root / "disappearing.vrm";
        const fs::path disappeared = root / "disappeared.vrm";
        fs::copy_file(valid, disappearing);
        const auto disappearing_contract = contract_for(disappearing, "fixture_a");
        const auto disappeared_model = product::load_verified_vrm_component(
            disappearing, disappearing_contract, nullptr, [&] {
                await_filesystem_tick(root, stat_file(disappearing).st_ctim);
                fs::rename(disappearing, disappeared);
            });
        require_test(!fs::exists(disappearing) &&
                         disappeared_model->architecture_id() == "fixture_a",
                     "pathname disappearance changed stable verification");

        const fs::path changed = root / "changed.vrm";
        fs::copy_file(valid, changed);
        const auto changed_contract = contract_for(changed, "fixture_a");
        expect_failure([&] {
            (void)product::load_verified_vrm_component(
                changed, changed_contract, nullptr, [&] {
                    change_byte(changed,
                        static_cast<std::streamoff>(fs::file_size(changed) - 1));
                });
        }, "in-place mutation");

        stable_namespace_tests(root);

        fs::remove_all(root);
        std::cout << "concurrent VRM verification tests: PASS\n";
        return 0;
    } catch (const std::exception& failure) {
        fs::remove_all(root);
        std::cerr << "concurrent VRM verification tests: FAIL: "
                  << failure.what() << '\n';
        return 1;
    }
}
