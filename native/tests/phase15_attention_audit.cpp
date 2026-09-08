#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "vrhino/backend/cuda_backend.h"
#include "vrhino/bundle.h"
#include "vrhino/error.h"

namespace {

struct Metrics {
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double difference_square = 0.0;
    double reference_square = 0.0;
    double actual_square = 0.0;
    double dot = 0.0;
    int64_t elements = 0;
    int64_t failing = 0;
    int64_t first_failing = -1;
};

void add(Metrics& result, float reference, float actual, int64_t index) {
    const double difference = std::abs(static_cast<double>(actual) - reference);
    result.max_abs = std::max(result.max_abs, difference);
    result.sum_abs += difference;
    result.difference_square += difference * difference;
    result.reference_square += static_cast<double>(reference) * reference;
    result.actual_square += static_cast<double>(actual) * actual;
    result.dot += static_cast<double>(reference) * actual;
    ++result.elements;
    if (!std::isfinite(actual) || difference > 5e-5 + 2e-5 * std::abs(reference)) {
        if (result.first_failing < 0) result.first_failing = index;
        ++result.failing;
    }
}

void print(const std::string& prefix, const Metrics& value) {
    const double relative_l2 = std::sqrt(
        value.difference_square / std::max(value.reference_square, 1e-300));
    const double cosine = value.dot / std::sqrt(
        std::max(value.reference_square * value.actual_square, 1e-300));
    std::cout << prefix << ".max_abs=" << value.max_abs << '\n'
              << prefix << ".mean_abs=" << value.sum_abs / value.elements << '\n'
              << prefix << ".relative_l2=" << relative_l2 << '\n'
              << prefix << ".cosine=" << cosine << '\n'
              << prefix << ".failing_element_count=" << value.failing << '\n'
              << prefix << ".first_failing_flat_index=" << value.first_failing << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vrhino::require(argc == 2 || argc == 3,
            "usage: vrhino-phase15-attention-audit INPUT.bundle [OUTPUT.bundle]");
        vrhino::TensorBundle capture = vrhino::read_bundle(argv[1]);
        const vrhino::Tensor& q = capture.at("q");
        const vrhino::Tensor& k = capture.at("k");
        const vrhino::Tensor& v = capture.at("v");
        const vrhino::Tensor& reference = capture.at("output.correct_bounded");
        vrhino::require(q.dtype() == vrhino::DType::F32 && q.shape() == k.shape() &&
                       k.shape() == v.shape() && q.shape() == reference.shape(),
                       "Phase 15 Attention audit expects matching FP32 BSHD tensors");

        vrhino::CudaBackend backend;
        backend.set_execution_dtype(vrhino::DType::F32);
        const vrhino::Tensor q_device = backend.copy_to_device(q, vrhino::DType::F32);
        const vrhino::Tensor k_device = backend.copy_to_device(k, vrhino::DType::F32);
        const vrhino::Tensor v_device = backend.copy_to_device(v, vrhino::DType::F32);
        backend.enable_profiling(true);
        const vrhino::Tensor output_device = backend.attention(
            q_device, k_device, v_device, nullptr,
            capture.at("causal").data_as<int64_t>()[0] != 0,
            capture.at("scale").data_as<float>()[0]);
        const vrhino::Tensor output = backend.copy_to_host(output_device);
        const auto profile = backend.profile_stats();

        Metrics overall;
        std::vector<Metrics> rows(static_cast<size_t>(q.dim(1)));
        for (int64_t index = 0; index < output.numel(); ++index) {
            const float expected = reference.data_as<float>()[index];
            const float actual = output.data_as<float>()[index];
            add(overall, expected, actual, index);
            const int64_t row = (index / q.dim(3) / q.dim(2)) % q.dim(1);
            add(rows[static_cast<size_t>(row)], expected, actual, index);
        }
        const auto worst = std::max_element(rows.begin(), rows.end(),
            [](const Metrics& a, const Metrics& b) { return a.max_abs < b.max_abs; });

        std::cout << std::setprecision(12)
                  << "q_tokens=" << q.dim(1) << '\n'
                  << "k_tokens=" << k.dim(1) << '\n'
                  << "heads=" << q.dim(2) << '\n'
                  << "head_dimension=" << q.dim(3) << '\n';
        for (const auto& [name, stat] : profile) {
            if (name.starts_with("attention"))
                std::cout << "attention_device_ms=" << stat.device_milliseconds << '\n';
        }
        print("overall", overall);
        print("row.first", rows.front());
        print("row.middle", rows[static_cast<size_t>(q.dim(1) / 2)]);
        print("row.last", rows.back());
        std::cout << "row.worst.index=" << std::distance(rows.begin(), worst) << '\n';
        print("row.worst", *worst);
        std::cout << "status=" << (overall.failing == 0 ? "PASS" : "FAIL") << '\n';

        if (argc == 3) vrhino::write_bundle(argv[2], {{"output", output}});
        return overall.failing == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "phase15 Attention audit: " << error.what() << '\n';
        return 1;
    }
}
