"""Compare the original fixed family graph declarations with the current builder.

Metadata/admission only: the baseline test runs no model or device execution.
All extracted baseline files and snapshots stay outside the repository.
"""
import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(args.output_dir).resolve()
    build = pathlib.Path(args.build_dir).resolve()
    if out == repo or repo in out.parents:
        raise SystemExit("Baseline output must be outside the repository")

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(repo), *parts])

    revision = git("rev-parse", "--verify", args.baseline + "^{commit}").decode().strip()
    for name in ["src/architectures/wan_self_attention_graph.h", "src/runtime/neural_graph.h",
                 "tests/wan_graph_dual_path_tests.cpp", "tests/neural_graph_test_backend.h"]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("show", revision + ":native/" + name))
    binary = out / "baseline-family-declaration"
    subprocess.run(["c++", "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                    "-I" + str(repo / "native/include"), str(out / "tests/wan_graph_dual_path_tests.cpp"),
                    "-Wl,--start-group", *[str(build / ("lib" + name + ".a")) for name in
                    ["vrhino_native_common", "vrhino_neural_graph", "vrhino_ltx_graph"]],
                    "-Wl,--end-group", "-o", str(binary)], check=True)
    before = subprocess.check_output([str(binary)])
    after = subprocess.check_output([str(build / "vrhino-wan-graph-dual-path-tests")])
    (out / "baseline.snapshot").write_bytes(before)
    (out / "current.snapshot").write_bytes(after)
    if before != after:
        raise SystemExit("FAIL: family declaration/admission snapshots differ")
    print(f"FAMILY_BASELINE_DECLARATION_EXACT=PASS baseline={revision} bytes={len(before)}")


if __name__ == "__main__":
    main()
