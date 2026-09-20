#pragma once
#include "vrhino/product/run.h"

namespace vrhino::product {
// Product-only lowering result. No Backend pointer or solver state; this does
// not grant numerical admission. Canonical programs retain sampling ownership.
struct DeclaredTextRun {
    Json request;
    MemoryBudget memory;
    std::string precision_artifact;
    int64_t fps = 0;
    float video_minimum = 0;
    float video_maximum = 0;
};
DeclaredTextRun lower_declared_text_run(const ResolvedRunnableModel&, const RunOptions&);
RunResult run_declared_text_product(const ResolvedRunnableModel&, const RunOptions&, RunEventSink);
} // namespace vrhino::product
