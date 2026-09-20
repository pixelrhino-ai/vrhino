"""Build a synthetic sampler snapshot against a Git baseline, compare exact bytes.

No weights, GPU, network, checkout or worktree mutation. Temporary baseline
headers/sources and the executable live only in --output-dir.
"""
import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current-executable", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(args.output_dir).resolve()
    if out == repo or repo in out.parents:
        raise SystemExit("Baseline output directory must be outside the source repository")
    out.mkdir(parents=True, exist_ok=True)

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(repo), *parts])

    revision = git("rev-parse", "--verify", args.baseline + "^{commit}").decode().strip()
    headers = git("ls-tree", "-r", "--name-only", revision, "native/include").decode().splitlines()
    sources = ["json.cpp", "tensor.cpp", "tensor_util.cpp", "precision/precision.cpp",
               "sampling/sampling.cpp", "runtime/prepared_tensor.cpp"]
    for name in headers + ["native/src/" + source for source in sources]:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git("show", revision + ":" + name))
    binary = out / "baseline-snapshot"
    subprocess.run(["c++", "-std=c++20", "-O0", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(out / "native/include"), "-I" + str(repo / "native/tests"),
                    str(repo / "native/tests/step_execution_legacy_snapshot.cpp"),
                    *[str(out / "native/src" / source) for source in sources],
                    "-o", str(binary)], check=True)
    before = subprocess.check_output([str(binary)])
    after = subprocess.check_output([str(pathlib.Path(args.current_executable).resolve())])
    (out / "baseline.snapshot").write_bytes(before)
    (out / "current.snapshot").write_bytes(after)
    if before != after:
        raise SystemExit("FAIL: baseline/current snapshots differ; inspect output directory")
    print(f"LEGACY_BASELINE_BYTE_EXACT=PASS baseline={revision} cases=16 bytes={len(before)}")


if __name__ == "__main__":
    main()
