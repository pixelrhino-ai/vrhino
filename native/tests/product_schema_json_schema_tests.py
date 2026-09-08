#!/usr/bin/env python3

import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parents[1])
    schema_path = source.parent / "docs/product/vrhino-model-v0.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    legacy = sorted((source / "specs").glob("*/vrhino-model.json"))
    successors = sorted((source / "specs").glob("*/successors/*/vrhino-model.json"))
    require(len(successors) == 5, "expected exactly five successor manifests")
    for path in legacy + successors:
        errors = sorted(
            validator.iter_errors(json.loads(path.read_text(encoding="utf-8"))),
            key=lambda error: list(error.path),
        )
        require(not errors, f"{path}: {errors[0].message if errors else ''}")

    ltx = json.loads(successors[0].read_text(encoding="utf-8"))
    invalid_version = copy.deepcopy(ltx)
    invalid_version["product"]["input_schema"]["schema"] = (
        "vrhino.product.input-schema.v2"
    )
    require(not validator.is_valid(invalid_version), "unknown schema version passed")

    invalid_type = copy.deepcopy(ltx)
    invalid_type["product"]["input_schema"]["inputs"][0]["type"] = "image"
    require(not validator.is_valid(invalid_type), "unknown Product type passed")

    invalid_semantic = copy.deepcopy(ltx)
    invalid_semantic["product"]["input_schema"]["inputs"][0]["validation"][
        "script"
    ] = "accept()"
    require(not validator.is_valid(invalid_semantic), "executable validation passed")

    duplicated_required_inputs = copy.deepcopy(ltx)
    duplicated_required_inputs["product"]["required_inputs"] = ["prompt"]
    require(
        not validator.is_valid(duplicated_required_inputs),
        "schema-backed required_inputs duplication passed",
    )

    print(
        f"package JSON Schema tests: PASS ({len(legacy)} legacy, "
        f"{len(successors)} successors)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"package JSON Schema tests: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
