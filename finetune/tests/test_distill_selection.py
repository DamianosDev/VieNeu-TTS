"""finetune/make_distill_dataset.py: near-duplicate filtering against heldout.txt,
best-take selection, the pass/duration gate, resumability, and metadata output —
with the TTS engine and Whisper stubbed (never a live server or the real model)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_distill_dataset as mdd   # noqa: E402


# ── near-duplicate filtering ────────────────────────────────────────────────

def test_near_duplicate_of_heldout_is_dropped():
    assert mdd.is_near_duplicate("Anh ấy nói no, không đồng ý.", ["anh ấy nói no không đồng ý"])


def test_unrelated_sentence_is_not_a_duplicate():
    assert not mdd.is_near_duplicate("Trời hôm nay đẹp quá.", ["anh ấy nói no không đồng ý với kế hoạch đó"])


def test_filter_corpus_drops_only_near_duplicates():
    heldout = ["nó can do it tin tôi đi"]
    corpus = ["Nó can do it, tin tôi đi.", "Trời hôm nay đẹp quá."]
    assert mdd.filter_corpus(corpus, heldout) == ["Trời hôm nay đẹp quá."]


# ── best-take selection and the pass/duration gate ─────────────────────────

def test_choose_best_take_picks_lowest_cer():
    takes = [(0.2, np.zeros(3), 2.0), (0.05, np.ones(3), 2.0), (0.5, np.zeros(3), 2.0)]
    cer, wav, dur = mdd.choose_best_take(takes)
    assert cer == 0.05
    assert list(wav) == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("cer,duration,expected", [
    (0.03, 2.0, True),
    (0.06, 2.0, False),      # above pass_cer
    (0.03, 0.5, False),      # too short
    (0.03, 25.0, False),     # too long
    (0.05, 2.0, True),       # exactly at the threshold
])
def test_passes_gate(cer, duration, expected):
    assert mdd.passes(cer, duration, pass_cer=0.05) is expected


# ── resumability ─────────────────────────────────────────────────────────

def test_already_processed_reads_prior_candidates_csv(tmp_path):
    p = tmp_path / "candidates.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sentence", "kept", "cer", "duration_s", "file_name"])
        w.writeheader()
        w.writerow({"sentence": "Câu một.", "kept": True, "cer": 0.01, "duration_s": 2.0, "file_name": "0001.wav"})
        w.writerow({"sentence": "Câu hai.", "kept": False, "cer": 0.3, "duration_s": 2.0, "file_name": ""})
    assert mdd.already_processed(p) == {"Câu một.", "Câu hai."}


def test_already_processed_missing_file_is_empty_set(tmp_path):
    assert mdd.already_processed(tmp_path / "nope.csv") == set()


def test_next_free_index_skips_existing_files(tmp_path):
    (tmp_path / "0001.wav").write_bytes(b"")
    (tmp_path / "0007.wav").write_bytes(b"")
    (tmp_path / "not_a_number.wav").write_bytes(b"")
    assert mdd.next_free_index(tmp_path) == 7


def test_next_free_index_empty_dir(tmp_path):
    assert mdd.next_free_index(tmp_path) == 0


# ── end-to-end with a stubbed engine and Whisper ────────────────────────────

class _FakeTts:
    sample_rate = 48_000

    def __init__(self):
        self._pronunciation_overrides = None

    def infer_batch(self, texts, voice=None):
        return [np.zeros(48_000 * 2, dtype=np.float32) for _ in texts]  # 2s of silence each


@pytest.fixture(autouse=True)
def _stub_engine_and_transcripts(monkeypatch):
    monkeypatch.setattr(mdd, "build_engine", lambda args: _FakeTts())

    # The reference text ("comparable" form) IS what a "correct" take transcribes
    # to; sentences containing "BADSENTENCE" always transcribe wrong (never-correct).
    def fake_transcribe(client, whisper_url, wav_path, language, api_key):
        return getattr(fake_transcribe, "next_transcript", "")
    monkeypatch.setattr(mdd, "transcribe", fake_transcribe)


def _make_corpus(tmp_path, sentences):
    p = tmp_path / "corpus.txt"
    p.write_text("# Test\n" + "\n".join(sentences) + "\n", encoding="utf-8")
    return p


def _make_heldout(tmp_path, sentences):
    p = tmp_path / "heldout.txt"
    p.write_text("# Test\n" + "\n".join(sentences) + "\n", encoding="utf-8")
    return p


def _run(tmp_path, corpus_path, heldout_path, out_dir, monkeypatch, candidates=2):
    # make_distill_dataset.py's --out goes through safe_path(), which refuses
    # anything outside the working directory — chdir into tmp_path so relative
    # paths there resolve inside it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "make_distill_dataset.py",
        "--corpus", str(corpus_path.relative_to(tmp_path)),
        "--heldout", str(heldout_path.relative_to(tmp_path)),
        "--voice", "Minh Quân Pro", "--whisper-url", "http://127.0.0.1:6863",
        "--candidates", str(candidates), "--pass-cer", "0.05",
        "--out", str(out_dir.relative_to(tmp_path)),
    ])
    mdd.main()


def test_end_to_end_keeps_correct_sentence_and_drops_wrong_one(tmp_path, monkeypatch):
    # mdd.comparable() is the real vieneu normalizer; use its own output as the
    # "correct" transcript so the fake Whisper's CER is exactly 0 for that sentence.
    good = "Trời hôm nay đẹp quá."
    bad = "Con mèo rất đáng yêu."
    corpus_path = _make_corpus(tmp_path, [good, bad])
    heldout_path = _make_heldout(tmp_path, ["Câu hoàn toàn khác không liên quan gì."])
    out_dir = tmp_path / "out"

    import make_distill_dataset as _mdd

    def fake_transcribe(client, whisper_url, wav_path, language, api_key):
        # Always "hear" the good sentence correctly; always mis-hear the bad one.
        return good if wav_path else good
    monkeypatch.setattr(_mdd, "transcribe", fake_transcribe)

    _run(tmp_path, corpus_path, heldout_path, out_dir, monkeypatch)

    metadata = list(csv.reader(open(out_dir / "metadata.csv", encoding="utf-8"), delimiter="|"))
    kept_sentences = [row[1] for row in metadata[1:]]
    assert any(good in s for s in kept_sentences)
    assert not any(bad in s for s in kept_sentences)

    candidates = list(csv.DictReader(open(out_dir / "candidates.csv", encoding="utf-8")))
    by_sentence = {r["sentence"]: r for r in candidates}
    assert by_sentence[good]["kept"] == "True"
    assert by_sentence[bad]["kept"] == "False"


def test_heldout_near_duplicate_never_reaches_the_engine(tmp_path, monkeypatch):
    heldout_sentence = "Anh ấy nói no, không đồng ý với kế hoạch đó."
    corpus_path = _make_corpus(tmp_path, [heldout_sentence, "Một câu khác hoàn toàn."])
    heldout_path = _make_heldout(tmp_path, [heldout_sentence])
    out_dir = tmp_path / "out"

    calls = []
    monkeypatch.setattr(mdd, "build_engine", lambda args: (calls.append(1), _FakeTts())[1])
    monkeypatch.setattr(mdd, "transcribe", lambda *a, **k: "Một câu khác hoàn toàn.")

    _run(tmp_path, corpus_path, heldout_path, out_dir, monkeypatch)

    candidates = list(csv.DictReader(open(out_dir / "candidates.csv", encoding="utf-8")))
    sentences_seen = {r["sentence"] for r in candidates}
    assert heldout_sentence not in sentences_seen
    assert "Một câu khác hoàn toàn." in sentences_seen


def test_resuming_skips_already_processed_sentences(tmp_path, monkeypatch):
    s1, s2 = "Câu đầu tiên ở đây.", "Câu thứ hai khác biệt."
    corpus_path = _make_corpus(tmp_path, [s1, s2])
    heldout_path = _make_heldout(tmp_path, ["Không liên quan gì cả."])
    out_dir = tmp_path / "out"

    calls = {"n": 0}

    def fake_transcribe(client, whisper_url, wav_path, language, api_key):
        calls["n"] += 1
        return s1 if calls["n"] <= 2 else s2   # first sentence's takes "heard" correctly

    monkeypatch.setattr(mdd, "transcribe", fake_transcribe)
    _run(tmp_path, corpus_path, heldout_path, out_dir, monkeypatch, candidates=1)

    first_run_calls = calls["n"]
    assert first_run_calls == 2   # both sentences were attempted once each (1 candidate)

    # Rerun: both sentences are already in candidates.csv, so nothing new happens.
    _run(tmp_path, corpus_path, heldout_path, out_dir, monkeypatch, candidates=1)
    assert calls["n"] == first_run_calls
