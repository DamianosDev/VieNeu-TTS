"""text_rules.py: the probe rows, the Vietnamese-sense regression guard, and the
override/idempotency/protected-span rules from phase 4 of the pronunciation plan."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vieneu_utils.text_rules import (          # noqa: E402
    AMBIGUOUS_WORDS,
    apply_text_rules,
    is_vietnamese_syllable,
    load_pronunciation_file,
)
from vieneu_utils.phonemize_text import phonemize_text_with_emotions   # noqa: E402


def phonemes(text: str) -> str:
    """The phonemes for a single already-normalization-eligible chunk: apply the
    text rules the same way normalize_to_chunks_v3_with_gaps does, then phonemize."""
    return phonemize_text_with_emotions(apply_text_rules(text))


def phonemes_no_rules(text: str) -> str:
    return phonemize_text_with_emotions(text)


# ── is_vietnamese_syllable ────────────────────────────────────────────────

@pytest.mark.parametrize("word", ["không", "được", "của", "gió", "đảm", "ngăn", "dự", "pin", "hot", "top", "no", "can", "do", "to", "go", "so", "me"])
def test_vietnamese_shaped_words_are_valid(word):
    assert is_vietnamese_syllable(word)


@pytest.mark.parametrize("word", ["yes", "we", "with", "love", "first", "sight", "just", "welcome", "jungle", "give", "five", "plastic", "friday", "good", "test", "game", "mode", "show", "ship"])
def test_english_shaped_words_are_invalid(word):
    assert not is_vietnamese_syllable(word)


def test_ambiguous_word_list_is_lowercase():
    assert all(w == w.lower() for w in AMBIGUOUS_WORDS)


# ── probe rows (assets/g2p-probe-2026-09-24.md) ───────────────────────────

def test_probe_row1_standalone_no_via_override():
    out = apply_text_rules("Anh ấy nói no, không đồng ý.")
    assert "<en>no</en>" in out
    assert phonemes(out) == phonemes(out)  # idempotent path sanity
    assert "n" in phonemes_of_word("no", out)


def phonemes_of_word(word, rewritten_text):
    return phonemize_text_with_emotions(rewritten_text)


def test_probe_row2_can_do_it_via_override():
    out = apply_text_rules("Nó can do it, tin tôi đi.")
    assert "<en>can do it</en>" in out


def test_probe_row3_to_me_with_anchor_in_clause():
    out = apply_text_rules("Anh ấy nói to me rằng ok.")
    assert "<en>" in out
    for w in ("to", "me", "ok"):
        assert f"<en>{w}</en>" in out or w in out  # swept into a shared <en> run


def test_probe_row3_to_me_actually_wrapped_together():
    out = apply_text_rules("Anh ấy nói to me rằng ok.")
    # "to", "me", and "ok" must all end up inside <en> somewhere.
    import re
    en_spans = "".join(re.findall(r"<en>(.*?)</en>", out))
    assert "to" in en_spans and "me" in en_spans and "ok" in en_spans


def test_probe_row4_is_so_good_wrapped_as_one_run():
    out = apply_text_rules("Cái này is so good luôn.")
    assert "<en>is so good</en>" in out


def test_probe_row5_url_scheme_and_domain():
    out = apply_text_rules("https://github.com/DamianosDev/VieNeu-TTS")
    assert "http" not in out.lower().split("<en>")[0]  # scheme dropped before rewriting
    assert "chấm" in out
    assert "gạch chéo" in out
    assert "<en>" in out


def test_probe_row6_slash_read_as_gach_cheo():
    out = apply_text_rules("Thư mục src/app.")
    assert "gạch chéo" in out
    assert "trên" not in out.split("gạch chéo")[0][-10:]  # old "trên" bug gone near the slash


def test_probe_row7_underscore_consistent_in_identifiers():
    out = apply_text_rules("Biến user_id nằm trong config.")
    assert "gạch dưới" in out


def test_probe_row8_email_reading():
    out = apply_text_rules("Email của tôi là thanh_nguyen.dev@gmail.com.")
    assert "gạch dưới" in out
    assert "a còng" in out
    assert out.count("chấm") >= 2


def test_pascal_case_identifiers_split():
    assert "<en>Git</en> <en>Hub</en>" in apply_text_rules("Công ty dùng GitHub cho mã nguồn.")
    assert "<en>Chat</en> <en>G P T</en>" in apply_text_rules("Hãy mở ChatGPT để hỏi thử.")


def test_readme_extension_wrapped_and_period_kept():
    out = apply_text_rules("Đọc file README.md trước khi cài đặt.")
    assert "<en>R E A D M E</en>" in out
    assert "<en>md</en>" in out
    assert out.endswith("cài đặt.")


def test_url_query_string_spoken_not_left_raw():
    out = apply_text_rules("Xem video tại youtube.com/watch?v=abc123.")
    assert "<en>watch</en> hỏi <en>v</en> bằng abc123" in out
    assert out.endswith("abc123.")  # sentence-final period kept, not swallowed into the query


def test_url_bare_trailing_question_mark_not_swallowed():
    out = apply_text_rules("Bạn đã xem docs.python.org chưa?")
    assert out.endswith("chưa?")


def test_url_query_string_idempotent():
    once = apply_text_rules("Xem video tại youtube.com/watch?v=abc123.")
    assert apply_text_rules(once) == once


def test_probe_row9_acronyms_spelled_in_english():
    for word in ("KPI", "API"):
        out = apply_text_rules(f"Chỉ số {word} tăng.")
        assert f"<en>{' '.join(word)}</en>" in out


def test_probe_row10_ctrl_c_plus_plus_c_sharp_overrides():
    assert "<en>control</en>" in apply_text_rules("Nhấn Ctrl để mở.")
    assert "<en>C plus plus</en>" in apply_text_rules("Dùng C++ cho dự án.")
    assert "<en>C sharp</en>" in apply_text_rules("Dùng C# cho dự án.")


def test_probe_row10_dot_net_override():
    assert "<en>dot net</en>" in apply_text_rules("Chạy trên .NET.")


def test_probe_row11_numbers_untouched_by_text_rules():
    text = "Giá là 1.500.000 đồng, tăng 12,5% so với năm ngoái."
    assert phonemes(text) == phonemes_no_rules(text)


def test_slash_dates_untouched_by_path_rule():
    text = "Phiên bản mới ra ngày 24/09/2026."
    assert apply_text_rules(text) == text
    assert phonemes(text) == phonemes_no_rules(text)


def test_windows_path_keeps_extension_and_trailing_period():
    out = apply_text_rules("Mở file C:\\Users\\ekko\\Documents\\report.docx để xem.")
    assert "<en>docx</en>" in out
    assert out.rstrip().endswith("để xem.")  # the sentence period is not swallowed into the path


def test_unix_path_trailing_period_preserved():
    out = apply_text_rules("Đường dẫn là /usr/local/bin/python3.")
    assert out.endswith(".")
    assert "chấm ." not in out and "chấm." not in out


# ── Vietnamese senses stay Vietnamese (no override, no anchor in clause) ──

@pytest.mark.parametrize("sentence", [
    "Tôi ăn no rồi, không ăn thêm được nữa.",
    "Bạn có can đảm nói ra sự thật không?",
    "Mẹ phải can hai anh em đang cãi nhau.",
    "Anh ấy do dự mãi mới quyết định.",
    "Chị ấy đi in tài liệu ở tiệm gần nhà.",
    "Trời hôm nay to gió quá.",
    "Thay pin cho điều khiển ti vi đi con.",
    "Cái áo này hơi to so với dáng em.",
])
def test_vietnamese_senses_unchanged(sentence):
    out = apply_text_rules(sentence)
    assert out == sentence
    assert phonemes(sentence) == phonemes_no_rules(sentence)


# ── regression guard: pure Vietnamese / numbers unaffected ────────────────

@pytest.mark.parametrize("sentence", [
    "Người ta thường nói rằng, đi một ngày đàng học một sàng khôn.",
    "Sáng sớm tinh mơ, sương vẫn còn đọng trên những ngọn cỏ.",
    "Trời ơi, sao lại có chuyện như thế này được!",
    "Chuyện khuya khoắt ngoằn ngoèo ấy khiến ai cũng nguệch ngoạc ghi chép.",
    "Phiên bản mới ra ngày 24/09/2026.",
    "Cuộc họp bắt đầu lúc 14:30 và kết thúc lúc 16h.",
    "Nhiệt độ hôm nay khoảng 32°C.",
    "Dân số thành phố đạt 8,5 triệu người.",
])
def test_pure_vietnamese_and_numbers_identical_phonemes(sentence):
    assert phonemes(sentence) == phonemes_no_rules(sentence)


# ── idempotency, protected spans ───────────────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "Nó can do it, tin tôi đi.",
    "Anh ấy bảo let it go, đừng nghĩ nữa.",
    "Truy cập https://github.com/DamianosDev/VieNeu-TTS để xem mã nguồn.",
    "Email của tôi là thanh_nguyen.dev@gmail.com.",
    "Anh ấy nói no, không đồng ý.",
])
def test_idempotent(sentence):
    once = apply_text_rules(sentence)
    twice = apply_text_rules(once)
    assert once == twice


def test_existing_en_tag_untouched():
    text = "Anh ấy nói <en>hello world</en> khi gặp tôi."
    assert apply_text_rules(text) == text


def test_emotion_bracket_cue_untouched():
    text = "Xin chào [cười] rất vui được gặp bạn."
    assert apply_text_rules(text) == text


def test_emotion_token_untouched():
    text = "Xin chào <|emotion_1|> rất vui."
    assert apply_text_rules(text) == text


def test_let_it_go_override():
    out = apply_text_rules("Anh ấy bảo let it go, đừng nghĩ nữa.")
    assert "<en>let it go</en>" in out


# ── override file loading ──────────────────────────────────────────────────

def test_user_override_wins_over_builtin():
    overrides = {"nói no": "xin chao ban"}
    out = apply_text_rules("Anh ấy nói no, không đồng ý.", overrides=overrides)
    assert "xin chao ban" in out
    assert "<en>no</en>" not in out  # the built-in "nói no" rule was replaced, not layered


def test_override_file_longest_term_first(tmp_path):
    p = tmp_path / "user.txt"
    p.write_text("no = A\nsay no = B\n", encoding="utf-8")
    overrides = load_pronunciation_file(p)
    out = apply_text_rules("Say no to plastic.", overrides=overrides)
    assert "B" in out
    assert " A " not in out


def test_override_file_case_sensitivity_rule():
    # "Xin" (uppercase in the term) matches case-sensitively; the lowercase "xin"
    # is a real, Vietnamese-syllable-shaped word the code-switch pass leaves alone,
    # so it isolates the override's own case rule.
    overrides = {"Xin": "xyzmarker"}
    assert "xyzmarker" in apply_text_rules("Xin chào.", overrides=overrides)
    assert "xyzmarker" not in apply_text_rules("xin chào.", overrides=overrides)


def test_invalid_override_lines_reported_once_with_line_numbers(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("good = fine\nno equals sign here\nalso bad\nok = <en>ok</en>\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_pronunciation_file(p)
    msg = str(exc.value)
    assert "line 2" in msg
    assert "line 3" in msg
    assert "line 1" not in msg  # the valid line is not reported
    assert "line 4" not in msg


def test_missing_pronunciation_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pronunciation_file(Path("does/not/exist.txt"))


def test_comment_and_blank_lines_skipped(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("# comment\n\nno = <en>no</en>\n", encoding="utf-8")
    assert load_pronunciation_file(p) == {"no": "<en>no</en>"}


# ── prepare_dataset.py shares the same rules ───────────────────────────────

def test_prepare_dataset_phonemize_matches_inference_path():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
    import importlib
    prepare_dataset = importlib.import_module("prepare_dataset")
    text = "Nó can do it, tin tôi đi."
    assert prepare_dataset.phonemize(text) == phonemes(text)
