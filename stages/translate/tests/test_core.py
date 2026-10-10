import json
from collections.abc import Callable

import pytest
from translate.core import (
    Segment,
    TranslationError,
    build_messages,
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
    return json.dumps({"translations": [{"id": i, "text": f"{prefix.capitalize()} {i}"} for i in ids]})


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
        "id": "seg_0003", "speaker": "SPK_00", "seconds": 1.5, "max_chars": 22, "text": "line 3",
    }  # fmt: skip
    assert "Brazilian Portuguese" in messages[0]["content"]


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("not json", "expected JSON"),
        ('{"translations": [{"id": "seg_0001", "text": "a"}]}', "missing ids: seg_0002"),
        (
            '{"translations": [{"id": "seg_0001", "text": "a"}, {"id": "seg_0002", "text": " "}]}',
            "empty",
        ),
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
