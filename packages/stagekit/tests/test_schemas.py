import copy
import json
from pathlib import Path
from typing import Any

import pytest

from stagekit import ContractError, read_json, validate, write_json

SCHEMAS = [
    "diarization/v1",
    "manifest/v1",
    "speakers/v1",
    "stage-probe/v1",
    "synthesis/v1",
    "transcript/v1",
    "translation/v1",
]

Examples = dict[str, dict[str, Any]]


def test_every_schema_file_has_an_example(examples: Examples) -> None:
    files = Path(__file__).parents[1] / "src" / "stagekit" / "schemas"
    on_disk = sorted(p.stem.replace(".", "/") for p in files.glob("*.json"))
    assert on_disk == SCHEMAS == sorted(examples)


@pytest.mark.parametrize("schema_id", SCHEMAS)
def test_examples_are_valid(examples: Examples, schema_id: str) -> None:
    validate(examples[schema_id], schema_id)


def test_schema_field_must_match(examples: Examples) -> None:
    with pytest.raises(ContractError, match="expected schema transcript/v1"):
        validate(examples["speakers/v1"], "transcript/v1")


def test_unknown_schema() -> None:
    doc = {"schema": "nope/v9"}
    with pytest.raises(ContractError, match="unknown schema"):
        validate(doc, "nope/v9")


def test_bad_segment_id(examples: Examples) -> None:
    doc = examples["transcript/v1"]
    doc["segments"][0]["id"] = "seg_1"
    with pytest.raises(ContractError, match="segments/0/id"):
        validate(doc, "transcript/v1")


def test_extra_property_rejected(examples: Examples) -> None:
    doc = examples["synthesis/v1"]
    doc["segments"][0]["extra"] = 1
    with pytest.raises(ContractError, match="Additional properties"):
        validate(doc, "synthesis/v1")


def test_absolute_path_rejected(examples: Examples) -> None:
    doc = examples["synthesis/v1"]
    doc["segments"][0]["path"] = "/abs/seg_0001.wav"
    with pytest.raises(ContractError):
        validate(doc, "synthesis/v1")


def test_duplicate_segment_ids(examples: Examples) -> None:
    doc = examples["translation/v1"]
    doc["segments"].append(copy.deepcopy(doc["segments"][0]))
    with pytest.raises(ContractError, match="duplicate segment id seg_0001"):
        validate(doc, "translation/v1")


def test_start_after_end(examples: Examples) -> None:
    doc = examples["transcript/v1"]
    doc["segments"][0]["start"] = 5.0
    with pytest.raises(ContractError, match="starts after it ends"):
        validate(doc, "transcript/v1")


def test_write_json_refuses_invalid_doc(examples: Examples, tmp_path: Path) -> None:
    doc = examples["manifest/v1"]
    doc["stages"]["extract"]["status"] = "weird"
    target = tmp_path / "manifest.json"
    with pytest.raises(ContractError):
        write_json(target, doc)
    assert not target.exists()


def test_write_then_read_round_trip(examples: Examples, tmp_path: Path) -> None:
    target = tmp_path / "out" / "transcript.json"
    write_json(target, examples["transcript/v1"])
    assert read_json(target, "transcript/v1") == examples["transcript/v1"]


def test_read_json_reports_path(examples: Examples, tmp_path: Path) -> None:
    target = tmp_path / "t.json"
    target.write_text(json.dumps({"schema": "transcript/v1"}))
    with pytest.raises(ContractError, match=str(target)):
        read_json(target, "transcript/v1")
    target.write_text("{not json")
    with pytest.raises(ContractError, match="cannot read JSON"):
        read_json(target, "transcript/v1")
