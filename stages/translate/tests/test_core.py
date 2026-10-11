import json
from collections.abc import Callable

import pytest
from translate.core import (
    Segment,
    TranslationError,
    budgets,
    build_messages,
    choose,
    parse_response,
    tidy,
    translate_all,
)


def segs(n: int) -> list[Segment]:
    return [
        Segment(f"seg_{i:04d}", i * 2.0, i * 2.0 + 1.5, "SPK_00", f"line {i}")
        for i in range(1, n + 1)
    ]


def answer(ids: list[str], prefix: str = "linha") -> str:
    return json.dumps(
        {"translations": [{"id": i, "candidates": [f"{prefix.capitalize()} {i}"]} for i in ids]}
    )


def requested_ids(messages: list[dict[str, str]]) -> list[str]:
    first_user = next(m for m in messages if m["role"] == "user")
    return [line["id"] for line in json.loads(first_user["content"])["translate"]]


def fake_chat(behaviour: Callable[[list[str], int], str]) -> tuple[Callable, list]:
    calls: list[list[dict[str, str]]] = []

    def chat(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return behaviour(requested_ids(messages), len(calls))

    return chat, calls


def test_messages_carry_timing_and_context() -> None:
    s = segs(5)
    messages = build_messages(s[2:4], [(s[1], "linha 2")], [s[4]], source="en", target="pt-BR")
    payload = json.loads(messages[1]["content"])
    assert payload["context_before"] == [
        {"speaker": "SPK_00", "text": "line 2", "translation": "linha 2"}
    ]
    assert payload["translate"][0] == {
        "id": "seg_0003", "speaker": "SPK_00", "seconds": 1.5, "max_syllables": 0, "text": "line 3",
    }  # fmt: skip
    assert "Brazilian Portuguese" in messages[0]["content"]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("not json", "expected JSON"),
        ('{"translations": [{"id": "seg_0001", "candidates": ["a"]}]}', "missing ids: seg_0002"),
        (
            '{"translations": [{"id": "seg_0001", "candidates": ["a"]},'
            ' {"id": "seg_0002", "candidates": [" "]}]}',
            "empty",
        ),
        ('{"translations": [{"id": "seg_0001", "text": "a"}]}', "expected JSON"),
    ],
)
def test_parse_response_errors(raw: str, error: str) -> None:
    with pytest.raises(TranslationError, match=error):
        parse_response(raw, ["seg_0001", "seg_0002"])


def test_batches_and_progress() -> None:
    chat, calls = fake_chat(lambda ids, n: answer(ids))
    progress: list[tuple[int, int]] = []
    s = segs(5)
    out = translate_all(
        chat, s, todo={x.id for x in s}, known={}, source="en", target="pt-BR",
        batch_size=2, context=1, retries=1, on_progress=lambda d, t: progress.append((d, t)),
    )  # fmt: skip
    assert out["seg_0005"] == "Linha seg_0005"
    assert len(calls) == 3
    assert progress == [(2, 5), (4, 5), (5, 5)]
    # The second batch sees the first batch's translation as context.
    payload = json.loads(calls[1][1]["content"])
    assert payload["context_before"][0]["translation"] == "Linha seg_0002"


def test_retry_with_error_feedback() -> None:
    chat, calls = fake_chat(lambda ids, n: "garbage" if n == 1 else answer(ids))
    out = translate_all(chat, segs(2), todo={"seg_0001", "seg_0002"}, known={}, source="en",
                        target="pt-BR", batch_size=5, context=0, retries=2)  # fmt: skip
    assert len(out) == 2
    assert "Invalid answer" in calls[1][-1]["content"]


def test_split_batch_when_retries_fail() -> None:
    def behaviour(ids: list[str], n: int) -> str:
        return answer(ids[:-1]) if len(ids) > 1 else answer(ids)  # always drops one id in batches

    chat, _ = fake_chat(behaviour)
    s = segs(4)
    out = translate_all(chat, s, todo={x.id for x in s}, known={}, source="en", target="pt-BR",
                        batch_size=4, context=0, retries=1)  # fmt: skip
    assert sorted(out) == [x.id for x in s]


def test_single_segment_failure_raises() -> None:
    chat, _ = fake_chat(lambda ids, n: "{}")
    with pytest.raises(TranslationError, match="seg_0001"):
        translate_all(chat, segs(1), todo={"seg_0001"}, known={}, source="en", target="pt-BR",
                      batch_size=1, context=0, retries=1)  # fmt: skip


def test_known_translations_are_not_requested() -> None:
    chat, calls = fake_chat(lambda ids, n: answer(ids))
    s = segs(3)
    known = {"seg_0001": "x", "seg_0003": "y"}
    out = translate_all(chat, s, todo={"seg_0002"}, known=known, source="en", target="pt-BR",
                        batch_size=5, context=1, retries=0)  # fmt: skip
    assert out == {"seg_0001": "x", "seg_0002": "Linha seg_0002", "seg_0003": "y"}
    assert requested_ids(calls[0]) == ["seg_0002"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("bem-vindos de volta. hoje vamos falar", "Bem-vindos de volta. Hoje vamos falar"),
        ("  é isso? sim! «ótimo»… então tá", "É isso? Sim! «Ótimo»… Então tá"),
        ("o iPhone custa 3.5 mil", "O iPhone custa 3.5 mil"),
        ('ele disse: "olá". "tchau"', 'Ele disse: "olá". "Tchau"'),
    ],
)
def test_tidy(raw: str, expected: str) -> None:
    assert tidy(raw) == expected


def test_genders_reach_the_payload() -> None:
    from translate.core import with_genders

    lines = [("SPK_00", "Thanks."), ("SPK_01", "Sure."), ("SPK_01", "Tired?")]
    segs = with_genders(
        [Segment(f"seg_{i:04d}", i, i + 1, spk, text) for i, (spk, text) in enumerate(lines, 1)],
        {"SPK_00": "male", "SPK_01": "female"},
    )
    assert [(s.speaker_gender, s.listener_gender) for s in segs] == [
        ("male", "unknown"), ("female", "male"), ("female", "male"),
    ]  # fmt: skip
    payload = json.loads(
        build_messages(segs[1:2], [], [], source="en", target="pt-BR")[1]["content"]
    )
    line = payload["translate"][0]
    assert line["speaker_gender"] == "female" and line["listener_gender"] == "male"
    unknown = json.loads(
        build_messages(segs[:1], [], [], source="en", target="pt-BR")[1]["content"]
    )
    assert "listener_gender" not in unknown["translate"][0]


def test_choose_prefers_the_longest_that_fits() -> None:
    cands = ["Hoje vamos mostrar um produto novo", "Hoje mostramos um produto", "Um produto"]
    assert choose(cands, 9) == "Hoje mostramos um produto"  # 9 syllables
    assert choose(cands, 100) == cands[0]
    assert choose(cands, 2) == "Um produto"  # nothing fits: the shortest
    assert choose(cands, 0) == cands[0]  # no budget


def test_budgets_use_the_gap_to_the_next_line() -> None:
    lines = [Segment("seg_0001", 0.0, 1.0, "A", "x"), Segment("seg_0002", 3.15, 4.0, "B", "y")]
    assert budgets(lines, syllables_per_second=5.0, gap=0.15) == {"seg_0001": 15, "seg_0002": 4}


def test_shorten_lines_carry_the_current_text() -> None:
    seg = Segment("seg_0001", 0, 2, "A", "Hello there", budget=6, shorten="Olá a todos vocês aí")
    line = json.loads(build_messages([seg], [], [], source="en", target="pt-BR")[1]["content"])
    assert line["translate"][0]["shorten"] == "Olá a todos vocês aí"
    assert line["translate"][0]["max_syllables"] == 6
