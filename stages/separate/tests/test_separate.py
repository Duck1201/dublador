import pytest
from separate.__main__ import stem_role


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("audio_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav", "vocals"),
        ("audio_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.wav", "accompaniment"),
        ("/abs/audio_(No Vocals)_htdemucs.wav", "accompaniment"),
        ("audio_(Other)_htdemucs.wav", "accompaniment"),
        ("audio_(Drums)_htdemucs.wav", None),
    ],
)
def test_stem_role(name: str, role: str | None) -> None:
    assert stem_role(name) == role
