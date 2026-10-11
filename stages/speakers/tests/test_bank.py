"""Bank-voice choice: same gender, closest pitch, no repeats until voices run out."""

from pathlib import Path

import numpy as np
import pytest
import soundfile
from speakers.bank import BankVoice, assign, scan

VOICES = [
    BankVoice("male_01", "male", 87.0),
    BankVoice("male_02", "male", 122.0),
    BankVoice("male_03", "male", 180.0),
    BankVoice("female_01", "female", 207.0),
    BankVoice("female_02", "female", 290.0),
]


def test_closest_pitch_of_the_same_gender() -> None:
    chosen = assign([("SPK_00", "male", 90.0), ("SPK_01", "female", 280.0)], VOICES)
    assert chosen == {"SPK_00": "male_01", "SPK_01": "female_02"}


def test_no_repeats_until_the_gender_runs_out() -> None:
    speakers = [("A", "male", 90.0), ("B", "male", 92.0), ("C", "male", 95.0), ("D", "male", 99.0)]
    chosen = assign(speakers, VOICES)
    assert [chosen[s] for s in "ABC"] == ["male_01", "male_02", "male_03"]
    assert chosen["D"] == "male_01"  # all used: the closest again


def test_unknown_gender_and_missing_pitch() -> None:
    assert assign([("X", "unknown", 200.0)], VOICES) == {"X": "female_01"}
    assert assign([("X", "male", None)], VOICES) == {"X": "male_01"}
    with pytest.raises(ValueError, match="empty"):
        assign([("X", "male", 100.0)], [])


def test_scan_reads_the_bank_on_disk(tmp_path: Path) -> None:
    for gender, n in (("male", "01"), ("female", "02")):
        folder = tmp_path / "voices" / "pt-BR" / gender
        folder.mkdir(parents=True)
        soundfile.write(str(folder / f"{n}.wav"), np.zeros(100, dtype=np.float32), 16000)
        (folder / f"{n}.txt").write_text("frase\n")
    (tmp_path / "voices" / "pt-BR" / "male" / "02.wav").write_bytes(b"")  # no .txt: skipped
    voices = scan(tmp_path, lambda wav: 100.0)
    assert [(v.id, v.gender) for v in voices] == [("male_01", "male"), ("female_02", "female")]
