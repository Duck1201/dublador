from pathlib import Path

import pytest

from stagekit import atomic_path, atomic_write_json, atomic_write_text, hash_file, hash_value


def test_atomic_write_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.json"
    atomic_write_json(target, {"x": "ã"})
    assert target.read_text() == '{\n  "x": "ã"\n}\n'
    assert list(target.parent.iterdir()) == [target]


def test_interrupted_write_keeps_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "vocals.wav"
    atomic_write_text(target, "old")
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        assert tmp.suffix == ".wav"
        assert tmp.parent == target.parent
        tmp.write_text("half")
        raise RuntimeError("killed")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_hash_value_ignores_key_order() -> None:
    assert hash_value({"a": 1, "b": [1, 2]}) == hash_value({"b": [1, 2], "a": 1})
    assert hash_value({"a": 1}) != hash_value({"a": 2})
    assert hash_value("x").startswith("sha256:")


def test_hash_file(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    expected = "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_file(path) == expected
