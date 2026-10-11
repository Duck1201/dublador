"""pt-BR syllable counting heuristic."""

import pytest
from translate.syllables import count


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("computador", 4),
        ("saúde", 3),
        ("saída", 3),
        ("queijo", 2),
        ("guitarra", 3),
        ("pão", 1),
        ("Oi, pessoal.", 3),  # pes-so-al: unaccented hiatus is undercounted
        ("machine learning", 5),  # English words are approximate
        ("", 0),
    ],
)
def test_count(text: str, expected: int) -> None:
    assert count(text) == expected


def test_digits_count_as_spoken() -> None:
    assert count("em 2026") == 1 + 8
