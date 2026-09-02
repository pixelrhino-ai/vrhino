#!/usr/bin/env python3
"""Validate the release-facing v0.6.0-alpha contract projection."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - qualification environment gate
    raise SystemExit("jsonschema is required to run this repository validator") from exc


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ID = "vrhino.product.input-schema.v1"
SUCCESSORS = {
    "vrhino/ltx-video-v0.9.1:1.1.1": {
        "path": "registry/models/vrhino/ltx-video-v0.9.1/1.1.1",
        "files": {
            "pull-plan.json": "49be991555b7c04451c7cf0c16b38914b622566c55729c738b13255945d69a95",
            "source-plan.json": "cc517ebcdbf2ed7af546382183ca24fb2f0687c40db2f327bdfa9ddf3be7bb70",
            "vrhino-model.json": "6c41babc7ae117e7815b48e18ae7538d8deb39b58c26812e19fbf49f3784aaa1",
        },
        "seed": 5703,
        "output": (704, 480, 121, 25),
        "sampling": (40, 3.0),
    },
    "vrhino/wan2.1-t2v-1.3b:1.0.1": {
        "path": "registry/models/vrhino/wan2.1-t2v-1.3b/1.0.1",
        "files": {
            "pull-plan.json": "9ebe030c8a102c60c13eae12e70e7196f39a7eed7bcf1ec84565f674fafead8b",
            "source-plan.json": "a66e41ad12db555877af296758b44f4e3859d0dbbf2ec7d78b1b71c54ba1667c",
            "vrhino-model.json": "78ffd5a79e1e76a7b938651455e5200cb6bc17e82e9b8d0c5ddbf2438edf3f17",
        },
        "seed": 5701,
        "output": (832, 480, 81, 16),
        "sampling": (50, 5.0),
    },
    "vrhino/mochi-1-preview:1.0.1": {
        "path": "registry/models/vrhino/mochi-1-preview/1.0.1",
        "files": {
            "pull-plan.json": "c34785c495213c1875ae1a44194dfcae1de9742315e05c673c2e0228a1df9e0e",
            "source-plan.json": "c8db01ee37de240e48d2774ea697ec5222252c1da69d31898acd518114034170",
            "vrhino-model.json": "ddb9f272d68e8db9252211e0dda773564ce3ede96ba7d404aaed85b54ad3e339",
        },
        "seed": 11001,
        "output": (848, 480, 163, 30),
        "sampling": (64, 6.0),
    },
    "vrhino/musetalk-v1.5:1.0.1": {
        "path": "registry/models/vrhino/musetalk-v1.5/1.0.1",
        "files": {
            "execution.json": "a5e1592fec66c5fb2a69b0e965e2dde8a7a4d45319535123712ca5c7b04d4a68",
            "pull-plan.json": "fc61cbd7981d7018cbd98e27d4c2afbe2fbdda3700fe83f2cd71c35f4954896c",
            "source-plan.json": "3c0df235319de308f190a24b1bab1cfada5fdfb84fcf172ca4c4ec3a33989118",
            "vrhino-model.json": "ecb1278de565a1d4805cf5f4407787d4fa08bc3a3c69c40e344c78eac0fd2b01",
        },
        "seed": 11001,
    },
    "vrhino/latentsync-1.6:1.0.1": {
        "path": "registry/models/vrhino/latentsync-1.6/1.0.1",
        "files": {
            "execution.json": "f867e628bf4d2d588e265f480c30d0c60e9727d2e341a4300cc7c3081ce85b81",
            "pull-plan.json": "f42fdeace38c498ea5304af01b4b4440f43145be426cd952137df268c390b41d",
            "source-plan.json": "468528d5fa4b465c81da681bb1d8505af339b1beeef8938afcb38db2afb24ca3",
            "vrhino-model.json": "1efd88634f0a3831ab4708064bfb17be0b4dd706e6f9ad6285ef42f624fb066a",
        },
        "seed": 1247,
    },
}
HISTORICAL = {
    "registry/models/vrhino/latentsync-1.6/1.0.0/execution.json": "855a9eb478bc39148818ee39f342edf686da27516c41ca243a7d50d0c969908e",
    "registry/models/vrhino/latentsync-1.6/1.0.0/pull-plan.json": "1ea9388e20142ddbacc29451e575cc8d7ff6c5a7f2cd549379c63f12dfbb8363",
    "registry/models/vrhino/latentsync-1.6/1.0.0/source-plan.json": "2510c9c32b40ee1865c3c2cd391a5025bc27817a58a22e0d4dea9ad490c04ade",
    "registry/models/vrhino/latentsync-1.6/1.0.0/vrhino-model.json": "83f90a08ba3ac1a0634be8ca6b01d8d048f9eebd74cafc739e7d64d3e637d555",
    "registry/models/vrhino/latentsync-1.6/1.0.0/workflow.json": "8d556d4173e4e69daaed45dbea0bebbacc975b499b7b52d592d44e82715d1312",
    "registry/models/vrhino/musetalk-v1.5/1.0.0/execution.json": "c770d7ecbfe3c6cf21617176ebdf054a3c021c2bfc787f024ef8d919a5ca3cde",
    "registry/models/vrhino/musetalk-v1.5/1.0.0/pull-plan.json": "8d8bf2ea1d4c9a3ddfbabebcda5546732d60129ae76ff4fec19ee9f844c645f9",
    "registry/models/vrhino/musetalk-v1.5/1.0.0/source-plan.json": "92ed908857607ba920d079bd7567b87b7366d126ccd726f59a40e814f6537032",
    "registry/models/vrhino/musetalk-v1.5/1.0.0/vrhino-model.json": "08cccfb542b4dc6c9d43783f3f4c2b2525838e63b7f3aa226f1cd136dbc7dbe3",
    "registry/models/vrhino/musetalk-v1.5/1.0.0/workflow.json": "4240b286ee1bbde6b356a9648a9fc4c18b50b691ee88b78a341606b7959ca92b",
}
ROUTES = {
    "GET /api/v1/version",
    "GET /api/v1/models",
    "GET /api/v1/models/{model}",
    "POST /api/v1/runs",
    "GET /api/v1/runs/{id}",
    "DELETE /api/v1/runs/{id}",
    "GET /api/v1/runs/{id}/events",
}


def load(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declarations(schema: dict, group: str) -> dict[str, dict]:
    return {entry["name"]: entry for entry in schema[group]}


def check_successors() -> None:
    package_schema = load(ROOT / "spec/vrhino-model-v0.schema.json")
    validator = jsonschema.Draft202012Validator(package_schema)
    for reference, expected in SUCCESSORS.items():
        directory = ROOT / expected["path"]
        assert set(path.name for path in directory.iterdir()) == set(expected["files"])
        for name, wanted in expected["files"].items():
            assert digest(directory / name) == wanted, f"projection drift: {reference}/{name}"

        manifest = load(directory / "vrhino-model.json")
        validator.validate(manifest)
        identity = manifest["identity"]
        assert f'{identity["namespace"]}/{identity["name"]}:{identity["version"]}' == reference
        for plan_name in ("source-plan.json", "pull-plan.json"):
            assert load(directory / plan_name)["model_reference"] == reference

        product = manifest["product"]
        schema = product["input_schema"]
        assert schema["schema"] == SCHEMA_ID
        params = declarations(schema, "parameters")
        outputs = declarations(schema, "outputs")
        assert params["seed"]["default"] == expected["seed"]
        assert outputs["output"] == {
            "name": "output",
            "type": "media.mp4",
            "required": False,
            "default": "output.mp4",
            "validation": {"parent_creatable_and_writable": True},
        }

        if product["family"] == "text_to_video":
            inputs = declarations(schema, "inputs")
            assert set(inputs) == {"prompt"} and inputs["prompt"]["required"] is True
            profile = product["frozen_profile"]
            output = profile["output"]
            assert (output["width"], output["height"], output["frames"], output["fps"]["numerator"]) == expected["output"]
            assert output["fps"]["denominator"] == 1
            assert (profile["sampling"]["steps"], profile["sampling"]["guidance_scale"]) == expected["sampling"]
        else:
            inputs = declarations(schema, "inputs")
            assert set(inputs) == {"video", "audio"}
            assert inputs["video"]["required"] is True and inputs["audio"]["required"] is True
            assert inputs["video"]["validation"]["fps"] == {"numerator": 25, "denominator": 1}
            assert inputs["audio"]["validation"]["minimum_duration_ms"] == 40
            assert product["frozen_profile"]["output"]["duration"] == "audio_derived"

    latent = load(ROOT / SUCCESSORS["vrhino/latentsync-1.6:1.0.1"]["path"] / "vrhino-model.json")
    assert latent["product"]["frozen_profile"]["sampling"] == {
        "method": "ddim", "prediction": "epsilon", "steps": 20,
        "guidance_scale": 1.5, "eta": 0.0,
    }
    assert latent["product"]["frozen_profile"]["temporal"] == {"chunk_frames": 16}
    public_names = {
        item["name"]
        for group in ("inputs", "parameters", "outputs")
        for item in latent["product"]["input_schema"][group]
    }
    assert public_names == {"video", "audio", "seed", "output"}


def check_history() -> None:
    for relative, wanted in HISTORICAL.items():
        assert digest(ROOT / relative) == wanted, f"historical package changed: {relative}"


def check_api() -> None:
    api_doc = (ROOT / "docs/api/native-api-v1.md").read_text(encoding="utf-8")
    documented = {
        re.sub(r" +", " ", match.group(0))
        for match in re.finditer(r"^(?:GET|POST|DELETE) +/api/v1/[^\s`]+$", api_doc, re.MULTILINE)
    }
    assert documented == ROUTES, f"documented routes differ: {sorted(documented)}"

    example_dir = ROOT / "docs/api/examples/native-api-v1"
    examples = sorted(path.name for path in example_dir.iterdir() if path.is_file())
    assert examples == [
        "model-detail-successor-response.json", "model-list-response.json",
        "progress.ndjson", "run-accepted-response.json", "run-request.json",
        "run-status-responses.json", "truncation-error.json", "version-response.json",
    ]
    for path in example_dir.glob("*.json"):
        load(path)
    for line in (example_dir / "progress.ndjson").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        assert "sequence" in event and "kind" in event
    assert load(example_dir / "version-response.json")["version"] == "v0.6.0-alpha"


def check_docs() -> None:
    references = set(SUCCESSORS)
    for filename in ("README.md", "README.zh-CN.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert references <= set(re.findall(r"vrhino/[a-zA-Z0-9._-]+:[0-9.]+", text))
        assert "v0.6.0-alpha" in text

    # Validate repository-relative Markdown links without following the network.
    for path in [ROOT / "README.md", ROOT / "README.zh-CN.md", *ROOT.glob("docs/**/*.md"), *ROOT.glob("spec/*.md")]:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            clean = target.split("#", 1)[0]
            if not clean or "://" in clean or clean.startswith("mailto:"):
                continue
            assert (path.parent / clean).resolve().exists(), f"broken link: {path.relative_to(ROOT)} -> {target}"


def main() -> int:
    check_successors()
    check_history()
    check_api()
    check_docs()
    print("PASS: Public v0.6.0-alpha contract projection")
    print("successors=5 manifests=5 examples=8 routes=7 historical_files=10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
