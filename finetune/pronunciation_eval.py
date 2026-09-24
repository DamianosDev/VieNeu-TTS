"""Measure how a VieNeu-TTS preset voice pronounces a held-out sentence set.

For each sentence, synthesizes ``--takes`` renditions directly with the SDK
(``Vieneu(mode="v3turbo", ...)``, not the HTTP server), transcribes each
through the Whisper server's OpenAI-compatible ``POST /v1/audio/transcriptions``,
and scores the character error rate (CER) between the transcript and a
normalized "spoken reference" — so the model's own text normalization
(numbers, symbols) is never counted as an error, only what it actually said
wrong. A sentence passes when a majority of its takes are at or under
``--pass-cer``.

    .venv\\Scripts\\python finetune\\pronunciation_eval.py ^
      --sentences finetune\\pronunciation\\heldout.txt ^
      --voice "Minh Quân Pro" [--backbone-repo <dir or repo>] ^
      [--pronunciations <file>] [--no-text-rules] ^
      --whisper-url http://127.0.0.1:6863 [--whisper-api-key ...] ^
      --takes 3 --run baseline

Writes ``finetune/output/eval/<run>/`` (not committed — the fork's .gitignore
already excludes ``finetune/output/``): ``wavs/<index>_take<n>.wav``,
``report.csv`` (one row per take), and ``summary.json`` (per-category and
overall mean CER and pass rate, plus every failing sentence with its
phonemes and per-take CERs, for tagging by hand).
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from vieneu_lora.utils import safe_path   # noqa: E402


# ── sentence set ──────────────────────────────────────────────────────────

def read_sentences(path: Path) -> list[tuple[str, str]]:
    """``(category, sentence)`` pairs. A line starting with ``#`` sets the
    current category (including the file's own leading comment lines, which
    are harmlessly overwritten before the first sentence); blank lines are
    skipped."""
    category = ""
    out: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            category = line.lstrip("#").strip()
            continue
        out.append((category, line))
    return out


# ── text normalization for comparison and phonemes ──────────────────────────

def normalized_text(text: str) -> str:
    """The spoken-form text the model actually reads (numbers, symbols,
    sentence packing) — the front end ``V3TurboVieNeuTTS.infer`` itself uses."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps
    chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=1_000_000)
    return " ".join(chunks)


def phonemes_of(text: str) -> str:
    """The IPA phoneme string for ``text``, the same path the model uses
    (``normalize_to_chunks_v3_with_gaps`` then ``phonemize_text_with_emotions``
    per chunk) — for the report, not for synthesis."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions
    chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=1_000_000)
    return " ".join(phonemize_text_with_emotions(c) for c in chunks).strip()


_TAG_RE = re.compile(r"</?en>")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def comparable(text: str) -> str:
    """Lowercase, NFC, no ``<en>`` tags, no punctuation, single-spaced — the
    form both the reference and Whisper's transcript are compared in, so
    ``github.com`` and "github chấm com" compare equal."""
    t = normalized_text(text)
    t = _TAG_RE.sub("", t)
    t = unicodedata.normalize("NFC", t).lower()
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


def is_english_only(text: str) -> bool:
    """No Vietnamese letters or tone marks: every character is ASCII."""
    return all(ord(ch) < 128 for ch in text)


# ── character error rate ─────────────────────────────────────────────────

def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over characters, divided by the reference length.
    ``0.0`` when both are empty; ``1.0`` when the reference is empty but the
    hypothesis is not."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    prev = list(range(len(hypothesis) + 1))
    for i, rc in enumerate(reference, 1):
        cur = [i] + [0] * len(hypothesis)
        for j, hc in enumerate(hypothesis, 1):
            cost = 0 if rc == hc else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(reference)


# ── Whisper ───────────────────────────────────────────────────────────────

def transcribe(client: httpx.Client, whisper_url: str, wav_path: Path, language: str,
                api_key: Optional[str]) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with open(wav_path, "rb") as f:
        response = client.post(
            f"{whisper_url.rstrip('/')}/v1/audio/transcriptions",
            data={"model": "whisper", "language": language, "response_format": "json", "temperature": "0"},
            files={"file": (wav_path.name, f, "audio/wav")},
            headers=headers,
            timeout=60.0,
        )
    response.raise_for_status()
    body = response.json()
    text = body.get("text") if isinstance(body, dict) else None
    return " ".join(str(text or "").split())


# ── engine ────────────────────────────────────────────────────────────────

def build_engine(args: argparse.Namespace):
    """A ``V3TurboVieNeuTTS`` instance for ``--voice``. ``--pronunciations`` and
    ``--no-text-rules`` are accepted here ahead of phase 4: until that phase
    adds the matching constructor parameters, they are noted and ignored
    rather than raising, so this tool does not need changing again then."""
    from vieneu import Vieneu
    from vieneu.v3turbo import V3TurboVieNeuTTS

    kw: dict = dict(backend="auto", device="auto")
    if args.backbone_repo:
        kw["backbone_repo"] = args.backbone_repo
    accepted = inspect.signature(V3TurboVieNeuTTS.__init__).parameters
    if args.pronunciations:
        if "pronunciations" in accepted:
            kw["pronunciations"] = args.pronunciations
        else:
            print("  note: --pronunciations given, but this build has no such parameter yet (phase 4); ignored")
    if args.no_text_rules:
        if "text_rules" in accepted:
            kw["text_rules"] = False
        else:
            print("  note: --no-text-rules given, but this build has no text_rules parameter yet (phase 4); ignored")
    return Vieneu(mode="v3turbo", **kw)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--backbone-repo", default=None)
    ap.add_argument("--pronunciations", default=None)
    ap.add_argument("--no-text-rules", action="store_true")
    ap.add_argument("--whisper-url", required=True)
    ap.add_argument("--whisper-api-key", default=None)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--pass-cer", type=float, default=0.10)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", default=None, help="default: finetune/output/eval")
    args = ap.parse_args()

    items = read_sentences(safe_path(args.sentences))
    print(f"{len(items)} sentences from {args.sentences}")

    out_root = safe_path(args.out_dir) if args.out_dir else (ROOT / "finetune" / "output" / "eval")
    run_dir = out_root / args.run
    wav_dir = run_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    required_passes = args.takes // 2 + 1

    tts = build_engine(args)
    client = httpx.Client()

    rows: list[dict] = []
    per_category: dict[str, list[bool]] = {}
    failing: list[dict] = []
    t0 = time.perf_counter()
    for idx, (category, sentence) in enumerate(items, 1):
        ref = comparable(sentence)
        phon = phonemes_of(sentence)
        language = "en" if is_english_only(sentence) else "vi"
        take_cers = []
        for take in range(1, args.takes + 1):
            wav = tts.infer(sentence, voice=args.voice)
            wav_path = wav_dir / f"{idx:03d}_take{take}.wav"
            tts.save(wav, wav_path)
            transcript = transcribe(client, args.whisper_url, wav_path, language, args.whisper_api_key)
            cer = char_error_rate(ref, comparable(transcript))
            take_cers.append(cer)
            rows.append({"index": idx, "category": category, "sentence": sentence, "take": take,
                         "cer": round(cer, 4), "transcript": transcript, "phonemes": phon})
        ok = sum(1 for c in take_cers if c <= args.pass_cer) >= required_passes
        per_category.setdefault(category, []).append(ok)
        if not ok:
            failing.append({"index": idx, "category": category, "sentence": sentence,
                            "phonemes": phon, "cers": [round(c, 4) for c in take_cers]})
        if idx % 10 == 0 or idx == len(items):
            print(f"  {idx}/{len(items)}  ({time.perf_counter() - t0:.0f}s)")

    report_path = run_dir / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "category", "sentence", "take", "cer", "transcript", "phonemes"])
        w.writeheader()
        w.writerows(rows)

    all_cers = [r["cer"] for r in rows]
    total_sentences = len(items) or 1
    summary = {
        "run": args.run,
        "voice": args.voice,
        "takes": args.takes,
        "pass_cer": args.pass_cer,
        "sentences": len(items),
        "mean_cer": round(sum(all_cers) / len(all_cers), 4) if all_cers else 0.0,
        "pass_rate": round(sum(ok for oks in per_category.values() for ok in oks) / total_sentences, 4),
        "categories": {
            cat: {"sentences": len(oks), "pass_rate": round(sum(oks) / len(oks), 4)}
            for cat, oks in per_category.items()
        },
        "failing": failing,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{report_path}")
    print(f"{summary_path}")
    print(f"mean CER {summary['mean_cer']:.3f}, pass rate {summary['pass_rate']:.1%}, "
          f"{len(failing)}/{len(items)} sentences failing")


if __name__ == "__main__":
    main()
