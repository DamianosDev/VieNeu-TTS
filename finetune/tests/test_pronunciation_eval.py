"""finetune/pronunciation_eval.py: sentence-set parsing, language choice,
comparison normalization, and CER — all pure functions, plus `transcribe`
against a stubbed HTTP transport (never a live server)."""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pronunciation_eval import (   # noqa: E402
    char_error_rate,
    comparable,
    is_english_only,
    read_sentences,
    transcribe,
)


# ── read_sentences ────────────────────────────────────────────────────────

def test_read_sentences_groups_by_category_and_skips_blanks(tmp_path):
    p = tmp_path / "set.txt"
    p.write_text(
        "# File-level comment, not a category\n"
        "# Also just a comment\n"
        "\n"
        "# Ambiguous words\n"
        "Anh ấy nói no.\n"
        "\n"
        "Cô ấy chỉ trả lời đúng một chữ no.\n"
        "# URLs\n"
        "Vào trang vnexpress.net.\n",
        encoding="utf-8",
    )
    assert read_sentences(p) == [
        ("Ambiguous words", "Anh ấy nói no."),
        ("Ambiguous words", "Cô ấy chỉ trả lời đúng một chữ no."),
        ("URLs", "Vào trang vnexpress.net."),
    ]


def test_read_sentences_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("# only comments\n\n", encoding="utf-8")
    assert read_sentences(p) == []


# ── is_english_only (language=en vs vi) ──────────────────────────────────

def test_is_english_only_true_for_pure_ascii():
    assert is_english_only("Can you do it by Friday?")


def test_is_english_only_false_with_vietnamese_diacritics():
    assert not is_english_only("Anh ấy nói no, không đồng ý.")


def test_is_english_only_false_for_a_single_diacritic():
    assert not is_english_only("Tôi")


# ── comparable (spoken-reference normalization) ──────────────────────────

def test_comparable_strips_en_tags_case_and_punctuation():
    assert comparable("Nó <en>can do it</en>, tin tôi đi.") == comparable("nó can do it tin tôi đi")


def test_comparable_expands_numbers_like_the_model_reads_them():
    # normalize_to_chunks_v3_with_gaps spells out digits/symbols before G2P;
    # the comparison must use that spoken form, not the raw digits.
    spoken = comparable("Giá là 100 đồng.")
    assert "100" not in spoken
    assert spoken  # non-empty: the number was rendered as words


def test_comparable_collapses_whitespace_and_is_idempotent():
    once = comparable("Mình   deploy lên server rồi check log giúp nhé.")
    assert once == comparable(once)
    assert "  " not in once


def test_comparable_keeps_vietnamese_diacritics_only_punctuation_is_dropped():
    out = comparable("Trời ơi, sao lại có chuyện như thế này được!")
    assert "ơ" in out and "," not in out and "!" not in out


# ── char_error_rate ───────────────────────────────────────────────────────

def test_cer_identical_strings_is_zero():
    assert char_error_rate("xin chao", "xin chao") == 0.0


def test_cer_both_empty_is_zero():
    assert char_error_rate("", "") == 0.0


def test_cer_empty_reference_nonempty_hypothesis_is_one():
    assert char_error_rate("", "abc") == 1.0


def test_cer_one_substitution_over_short_reference():
    assert char_error_rate("cat", "cot") == pytest.approx(1 / 3)


def test_cer_counts_insertions_and_deletions():
    # "can" -> "canxyz": 3 insertions over a 3-char reference.
    assert char_error_rate("can", "canxyz") == pytest.approx(1.0)
    # "canxyz" -> "can": 3 deletions over a 6-char reference.
    assert char_error_rate("canxyz", "can") == pytest.approx(0.5)


# ── transcribe (stubbed HTTP client, never a live server) ────────────────

def test_transcribe_posts_language_and_returns_normalized_text(tmp_path):
    wav = tmp_path / "take1.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"text": "  xin   chào\nbạn  "})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    text = transcribe(client, "http://127.0.0.1:6863", wav, "vi", None)

    assert text == "xin chào bạn"
    assert seen["url"] == "http://127.0.0.1:6863/v1/audio/transcriptions"
    assert "authorization" not in seen["headers"]


def test_transcribe_sends_bearer_token_when_an_api_key_is_given(tmp_path):
    wav = tmp_path / "take1.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"text": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transcribe(client, "http://127.0.0.1:6863/", wav, "en", "secret-key")

    assert seen["authorization"] == "Bearer secret-key"


def test_transcribe_raises_on_server_error(tmp_path):
    wav = tmp_path / "take1.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        transcribe(client, "http://127.0.0.1:6863", wav, "vi", None)
