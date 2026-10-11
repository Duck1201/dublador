"""Rough pt-BR syllable count for length budgets (docs/RESEARCH.md §3.5).

Spoken length tracks syllables better than characters. The count only needs
to be consistent, since the speech rate it is divided by is calibrated with the
same function (`syllables_per_second`).
"""

import re

_VOWELS = "aeiouáéíóúâêôãõàüy"
_GROUP = re.compile(f"[{_VOWELS}]+")
_SILENT_U = re.compile(r"(?<=[qg])u(?=[eiéíêy])")  # que, qui, gue, gui
_NUMBER = re.compile(r"\d+")


def _word(word: str) -> int:
    word = _SILENT_U.sub("", word)
    count = 0
    for group in _GROUP.findall(word):
        count += 1
        # A vowel group is one syllable (diphthongs: pão, mãe, queijo), except
        # where an accent marks a hiatus: sa-ú-de, sa-í-da. Unaccented hiatus
        # (po-e-ta) and English words are undercounted; the calibrated rate
        # absorbs that.
        count += sum(ch in "íú" for ch in group[1:]) + (len(group) > 1 and group[0] in "íú")
    return count


def count(text: str) -> int:
    total = 0
    for token in re.findall(r"[\wáéíóúâêôãõàüç]+", text.lower()):
        digits = _NUMBER.fullmatch(token)
        total += 2 * len(token) if digits else _word(token)
    return total
