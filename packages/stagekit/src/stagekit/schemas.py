"""Versioned JSON Schemas of the file contract and validated JSON I/O."""

import json
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from stagekit.fileio import atomic_write_json


class ContractError(Exception):
    """A file does not follow the contract (schema, ids or times)."""


@cache
def _validator(schema_id: str) -> Draft202012Validator:
    name, _, version = schema_id.partition("/")
    resource = resources.files("stagekit") / "schemas" / f"{name}.{version}.json"
    if not version or not resource.is_file():
        raise ContractError(f"unknown schema: {schema_id}")
    schema = json.loads(resource.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate(doc: Any, schema_id: str) -> None:
    """Validate `doc` against `schema_id` (e.g. "transcript/v1") plus checks
    JSON Schema cannot express: unique segment ids and start <= end."""
    if not isinstance(doc, dict) or doc.get("schema") != schema_id:
        found = doc.get("schema") if isinstance(doc, dict) else type(doc).__name__
        raise ContractError(f"expected schema {schema_id}, found {found!r}")
    try:
        _validator(schema_id).validate(doc)
    except ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ContractError(f"{schema_id}: {where}: {e.message}") from e

    segments = doc.get("segments")
    if isinstance(segments, list):
        seen: set[str] = set()
        for seg in segments:
            seg_id = seg["id"]
            if seg_id in seen:
                raise ContractError(f"{schema_id}: duplicate segment id {seg_id}")
            seen.add(seg_id)
            if "start" in seg and "end" in seg and seg["start"] > seg["end"]:
                raise ContractError(f"{schema_id}: {seg_id} starts after it ends")


def read_json(path: Path, schema_id: str) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ContractError(f"{path}: cannot read JSON: {e}") from e
    try:
        validate(doc, schema_id)
    except ContractError as e:
        raise ContractError(f"{path}: {e}") from e
    return doc


def write_json(path: Path, doc: dict[str, Any]) -> None:
    """Validate, then write atomically. Invalid output never reaches the disk."""
    validate(doc, doc.get("schema", ""))
    atomic_write_json(path, doc)
