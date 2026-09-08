#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SUCCESSORS = [
    (
        "ltx_v0_9_1/successors/1.1.1/vrhino-model.json",
        "text_to_video",
        ["prompt"],
        5703,
        {"width": 704, "height": 480, "frames": 121, "fps": 25,
         "steps": 40, "guidance": 3.0},
    ),
    (
        "wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json",
        "text_to_video",
        ["prompt"],
        5701,
        {"width": 832, "height": 480, "frames": 81, "fps": 16,
         "steps": 50, "guidance": 5.0},
    ),
    (
        "mochi_1_preview/successors/1.0.1/vrhino-model.json",
        "text_to_video",
        ["prompt"],
        11001,
        {"width": 848, "height": 480, "frames": 163, "fps": 30,
         "steps": 64, "guidance": 6.0},
    ),
    (
        "public_musetalk_v15/successors/1.0.1/vrhino-model.json",
        "lip_sync",
        ["video", "audio"],
        11001,
        {"fps": 25},
    ),
    (
        "public_latentsync_16/successors/1.0.1/vrhino-model.json",
        "lip_sync",
        ["video", "audio"],
        1247,
        {"fps": 25, "steps": 20, "guidance": 1.5},
    ),
]

LEGACY = [
    "ltx_v0_9_1/vrhino-model.json",
    "wan2_1_t2v_1_3b/vrhino-model.json",
    "mochi_1_preview/vrhino-model.json",
    "public_musetalk_v15/vrhino-model.json",
    "public_latentsync_16/vrhino-model.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def reference(manifest: dict) -> str:
    identity = manifest["identity"]
    return f"{identity['namespace']}/{identity['name']}:{identity['version']}"


def install_sparse_fixture(cache: Path, manifest_path: Path) -> tuple[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["identity"]
    installed = (
        cache / "models" / identity["namespace"] / identity["name"] /
        identity["version"] / "vrhino-model.json"
    )
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, installed)
    for artifact in manifest["artifacts"]:
        sha256 = artifact["sha256"]
        blob = cache / "blobs" / "sha256" / sha256[:2] / sha256
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            blob.touch()
            os.truncate(blob, artifact["size"])
        require(blob.stat().st_size == artifact["size"], "shared blob size drift")
    return reference(manifest), manifest


def install_manifest_only(cache: Path, manifest: dict) -> str:
    identity = manifest["identity"]
    installed = (
        cache / "models" / identity["namespace"] / identity["name"] /
        identity["version"] / "vrhino-model.json"
    )
    installed.parent.mkdir(parents=True, exist_ok=True)
    installed.write_text(json.dumps(manifest, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    return reference(manifest)


def invoke(cli: Path, cache: Path, *arguments: str, timeout: int = 10):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [str(cli), "--cache-root", str(cache), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def declarations(schema: dict, group: str) -> dict[str, dict]:
    return {item["name"]: item for item in schema[group]}


def verify_successor(document: dict, manifest: dict, family: str,
                     required_inputs: list[str], seed: int, frozen: dict) -> None:
    require(document["schema_version"] == 1, "envelope version drift")
    require(document["model"]["reference"] == reference(manifest),
            "model reference drift")
    require(document["model"]["namespace"] == manifest["identity"]["namespace"],
            "structured identity drift")
    product = document["product"]
    require(product["family"] == family, "Product family drift")
    require(product["input_schema"] == manifest["product"]["input_schema"],
            "CLI maintains or mutates a second Product schema")
    require(product["frozen_profile"] == manifest["product"]["frozen_profile"],
            "CLI mutates frozen profile")
    schema = product["input_schema"]
    inputs = declarations(schema, "inputs")
    parameters = declarations(schema, "parameters")
    outputs = declarations(schema, "outputs")
    require([item["name"] for item in schema["inputs"] if item["required"]] ==
            required_inputs, "required input order/semantics drift")
    require(parameters["seed"]["default"] == seed, "seed default drift")
    require(parameters["seed"]["validation"]["maximum"] ==
            "18446744073709551615", "uint64 bound lost precision")
    require(outputs["output"]["required"] is False and
            outputs["output"]["default"] == "output.mp4",
            "optional output/default drift")
    profile = product["frozen_profile"]
    require(profile["output"]["fps"] ==
            {"numerator": frozen["fps"], "denominator": 1},
            "exact rational FPS drift")
    if family == "text_to_video":
        require(inputs["prompt"]["validation"]["min_length"] == 1,
                "prompt validation drift")
        require(profile["output"]["width"] == frozen["width"] and
                profile["output"]["height"] == frozen["height"] and
                profile["output"]["frames"] == frozen["frames"] and
                profile["sampling"]["steps"] == frozen["steps"] and
                profile["sampling"]["guidance_scale"] == frozen["guidance"],
                "TTV frozen facts drift")
        require(document["qualification"], "TTV qualification separation missing")
    else:
        require(inputs["video"]["validation"]["fps"] ==
                {"numerator": 25, "denominator": 1},
                "lip video validation drift")
        require(inputs["audio"]["validation"]["minimum_duration_ms"] == 40,
                "lip audio validation drift")
        require(profile["output"]["duration"] == "audio_derived",
                "lip duration semantics drift")
    require(document["installation"] == {"installed": True},
            "installation state drift")
    require(document["artifacts"]["declared_count"] == len(manifest["artifacts"]),
            "artifact count drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--require-cpu-binary", action="store_true")
    arguments = parser.parse_args()
    cli = arguments.cli.resolve()
    source_root = arguments.source_root.resolve()
    temporary = None
    if arguments.work_root:
        work = arguments.work_root.resolve()
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="vrhino-info-json-")
        work = Path(temporary.name)
    cache = work / "cache"

    if arguments.require_cpu_binary:
        linked = subprocess.run(
            ["ldd", str(cli)], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        ).stdout.lower()
        for library in ("libcuda", "libcublas", "libcudnn"):
            require(library not in linked, f"CPU-only CLI links {library}")

    installed_successors = []
    for relative, family, required_inputs, seed, frozen in SUCCESSORS:
        model_reference, manifest = install_sparse_fixture(
            cache, source_root / "specs" / relative)
        installed_successors.append(
            (model_reference, manifest, family, required_inputs, seed, frozen))
    installed_legacy = [
        install_sparse_fixture(cache, source_root / "specs" / relative)
        for relative in LEGACY
    ]

    expected_top_level = {
        "admission", "artifacts", "compatibility", "distribution",
        "installation", "license", "model", "product", "schema_version", "source",
    }
    for model_reference, manifest, family, required_inputs, seed, frozen in installed_successors:
        first = invoke(cli, cache, "info", model_reference, "--json")
        second = invoke(cli, cache, "info", model_reference, "--json")
        require(first.returncode == 0 and second.returncode == 0,
                f"info JSON failed: {model_reference}: {first.stderr}")
        require(first.stderr == second.stderr == "", "success wrote to stderr")
        require(first.stdout == second.stdout, "repeated CLI JSON differs")
        require(first.stdout.startswith("{") and first.stdout.endswith("}\n") and
                first.stdout.count("\n") == 1,
                "stdout contains prose or unstable pretty formatting")
        document = json.loads(first.stdout)
        require(set(document) in (expected_top_level,
                                  expected_top_level | {"qualification"}),
                "unexpected top-level JSON fields")
        verify_successor(document, manifest, family, required_inputs, seed, frozen)
        lowered = first.stdout.lower()
        for forbidden in (
            str(source_root).lower(), str(cache).lower(), "/nonpublic-test-home",
            "authorization:", "hf_token", "private/cas/path",
        ):
            require(forbidden not in lowered, f"privacy leak in JSON: {forbidden}")

    for model_reference, _manifest in installed_legacy:
        result = invoke(cli, cache, "info", model_reference, "--json")
        require(result.returncode == 0 and result.stderr == "",
                f"legacy info JSON failed: {model_reference}")
        document = json.loads(result.stdout)
        require(document["product"]["input_schema"] is None and
                document["product"]["frozen_profile"] is None,
                "legacy package was silently upgraded")

    ltx_reference = installed_successors[0][0]
    human = invoke(cli, cache, "info", ltx_reference)
    require(human.returncode == 0 and human.stderr == "" and
            human.stdout.startswith(f"Identity: {ltx_reference}\n") and
            "Manifest: " in human.stdout and not human.stdout.startswith("{"),
            "human-readable info behavior regressed")

    offline = subprocess.run(
        [str(cli), "--cache-root", str(cache), "--registry",
         "https://127.0.0.1:1", "info", ltx_reference, "--json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=3, check=False,
    )
    require(offline.returncode == 0 and json.loads(offline.stdout),
            "info unexpectedly used the configured network registry")

    missing = invoke(cli, cache, "info", "vrhino/not-installed:1.0.0", "--json")
    require(missing.returncode != 0 and missing.stdout == "" and missing.stderr,
            "failure emitted partial JSON or no stderr diagnostic")
    unsupported = invoke(cli, cache, "list", "--json")
    require(unsupported.returncode == 2 and unsupported.stdout == "" and
            unsupported.stderr.startswith("Usage:"),
            "--json was accepted outside info")

    invalid_version = json.loads(json.dumps(installed_successors[0][1]))
    invalid_version["identity"]["version"] = "invalid-info-schema"
    invalid_version["product"]["input_schema"]["schema"] = (
        "vrhino.product.input-schema.v2"
    )
    invalid_version_reference = install_manifest_only(cache, invalid_version)
    invalid = invoke(cli, cache, "info", invalid_version_reference, "--json")
    require(invalid.returncode != 0 and invalid.stdout == "" and
            "unsupported schema identity" in invalid.stderr,
            "unknown Product schema did not fail closed before JSON output")

    invalid_type = json.loads(json.dumps(installed_successors[0][1]))
    invalid_type["identity"]["version"] = "invalid-info-type"
    invalid_type["product"]["input_schema"]["inputs"][0]["type"] = "image"
    invalid_type_reference = install_manifest_only(cache, invalid_type)
    invalid = invoke(cli, cache, "info", invalid_type_reference, "--json")
    require(invalid.returncode != 0 and invalid.stdout == "" and
            "unsupported input type" in invalid.stderr,
            "unknown Product type did not fail closed before JSON output")

    latent_reference = installed_successors[-1][0]
    latent = invoke(cli, cache, "info", latent_reference, "--json").stdout.lower()
    for forbidden in (
        "alignment", "detector", '"tta"', "tensor", '"rng"',
    ):
        require(forbidden not in latent, f"LatentSync internal leaked: {forbidden}")

    if temporary is not None:
        temporary.cleanup()
    print("info JSON CLI consumer tests: PASS (5 successors, 5 legacy, CPU-only)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"info JSON CLI consumer tests: FAIL: {error}", file=os.sys.stderr)
        raise SystemExit(1)
