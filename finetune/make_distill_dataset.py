"""Generate a self-distilled, single-speaker training set for VieNeu-TTS, in
Minh Quân Pro's own voice, with the phase 4 text rules on.

For each corpus sentence (skipping anything close to a `heldout.txt` sentence — that
set must never leak into training), synthesizes ``--candidates`` takes in one GPU
batch, transcribes each through Whisper, and keeps the lowest-CER take when it is at
most ``--pass-cer`` and between 1 and 20 seconds long. Everything else is dropped and
logged in ``candidates.csv`` as ``never-correct`` — those need a pronunciation
override, not training. Output layout matches what ``finetune/prepare_dataset.py``
expects: ``metadata.csv`` (``file_name|text``) plus ``raw_audio/``.

    .venv\\Scripts\\python finetune\\make_distill_dataset.py ^
      --corpus finetune\\pronunciation\\distill-corpus.txt ^
      --heldout finetune\\pronunciation\\heldout.txt ^
      --voice "Minh Quân Pro" [--pronunciations <user file>] ^
      --whisper-url http://127.0.0.1:6863 [--whisper-api-key ...] ^
      --candidates 4 --pass-cer 0.05 --out finetune\\dataset\\minh_quan_pro

Resumable: rerunning skips every sentence already recorded in ``candidates.csv``
(kept or ``never-correct``), so a partial run can continue where it left off.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from vieneu_lora.utils import safe_path   # noqa: E402
from pronunciation_eval import (          # noqa: E402
    build_engine, char_error_rate, comparable, is_english_only, read_sentences, transcribe,
)

MIN_DURATION_S = 1.0
MAX_DURATION_S = 20.0
NEAR_DUP_RATIO = 0.8   # difflib.SequenceMatcher.ratio() over the comparable() form


def is_near_duplicate(sentence: str, heldout_comparable: Sequence[str]) -> bool:
    """Whether ``sentence`` is close enough to a held-out sentence to leak eval data
    into training — compared in the same normalized form CER uses, so ``no`` vs
    ``no,`` or different casing don't create false negatives."""
    c = comparable(sentence)
    return any(difflib.SequenceMatcher(None, c, h).ratio() >= NEAR_DUP_RATIO for h in heldout_comparable)


def filter_corpus(corpus: Sequence[str], heldout: Sequence[str]) -> List[str]:
    heldout_comparable = [comparable(h) for h in heldout]
    return [s for s in corpus if not is_near_duplicate(s, heldout_comparable)]


def choose_best_take(
    takes: Sequence[Tuple[float, np.ndarray, float]],
) -> Tuple[float, np.ndarray, float]:
    """``takes`` is ``[(cer, wav, duration_s), ...]``; returns the lowest-CER one."""
    return min(takes, key=lambda t: t[0])


def passes(cer: float, duration_s: float, pass_cer: float) -> bool:
    return cer <= pass_cer and MIN_DURATION_S <= duration_s <= MAX_DURATION_S


def already_processed(candidates_path: Path) -> set:
    """Sentences already logged in ``candidates.csv`` (kept or dropped) — resumability."""
    if not candidates_path.is_file():
        return set()
    with open(candidates_path, encoding="utf-8") as f:
        return {row["sentence"] for row in csv.DictReader(f)}


def next_free_index(audio_dir: Path) -> int:
    existing = [int(p.stem) for p in audio_dir.glob("*.wav") if p.stem.isdigit()]
    return max(existing, default=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--backbone-repo", default=None)
    ap.add_argument("--pronunciations", default=None)
    ap.add_argument("--whisper-url", required=True)
    ap.add_argument("--whisper-api-key", default=None)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--pass-cer", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = safe_path(args.out)
    audio_dir = out_dir / "raw_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"
    candidates_path = out_dir / "candidates.csv"

    heldout = [s for _, s in read_sentences(safe_path(args.heldout))]
    corpus = [s for _, s in read_sentences(safe_path(args.corpus))]
    sentences = filter_corpus(corpus, heldout)
    print(f"{len(corpus)} corpus sentences, {len(corpus) - len(sentences)} dropped as heldout "
          f"near-duplicates, {len(sentences)} left")

    done = already_processed(candidates_path)
    todo = [s for s in sentences if s not in done]
    print(f"{len(done)} already processed, {len(todo)} to go")
    if not todo:
        return

    tts = build_engine(argparse.Namespace(
        backbone_repo=args.backbone_repo, pronunciations=args.pronunciations, no_text_rules=False,
    ))
    from vieneu_utils.text_rules import apply_text_rules
    overrides = getattr(tts, "_pronunciation_overrides", None)

    client = httpx.Client()
    metadata_is_new = not metadata_path.is_file()
    candidates_is_new = not candidates_path.is_file()
    metadata_f = open(metadata_path, "a", newline="", encoding="utf-8")
    metadata_w = csv.writer(metadata_f, delimiter="|")
    if metadata_is_new:
        metadata_w.writerow(["file_name", "text"])
    candidates_f = open(candidates_path, "a", newline="", encoding="utf-8")
    candidates_w = csv.DictWriter(
        candidates_f, fieldnames=["sentence", "kept", "cer", "duration_s", "file_name"]
    )
    if candidates_is_new:
        candidates_w.writeheader()

    next_index = next_free_index(audio_dir)
    kept = 0
    scratch = audio_dir / "_scratch_candidate.wav"
    t0 = time.perf_counter()
    try:
        for i, sentence in enumerate(todo, 1):
            rewritten = apply_text_rules(sentence, overrides=overrides)
            ref = comparable(sentence)
            language = "en" if is_english_only(sentence) else "vi"
            wavs = tts.infer_batch([rewritten] * args.candidates, voice=args.voice)

            takes: List[Tuple[float, np.ndarray, float]] = []
            for wav in wavs:
                duration = len(wav) / tts.sample_rate
                sf.write(str(scratch), wav, tts.sample_rate)
                transcript = transcribe(client, args.whisper_url, scratch, language, args.whisper_api_key)
                cer = char_error_rate(ref, comparable(transcript))
                takes.append((cer, wav, duration))

            cer, wav, duration = choose_best_take(takes)
            file_name = ""
            if passes(cer, duration, args.pass_cer):
                next_index += 1
                file_name = f"{next_index:04d}.wav"
                sf.write(str(audio_dir / file_name), wav, tts.sample_rate)
                metadata_w.writerow([file_name, rewritten])
                metadata_f.flush()
                kept += 1
            candidates_w.writerow({
                "sentence": sentence, "kept": bool(file_name), "cer": round(cer, 4),
                "duration_s": round(duration, 2), "file_name": file_name,
            })
            candidates_f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  kept {kept}  ({time.perf_counter() - t0:.0f}s)")
    finally:
        metadata_f.close()
        candidates_f.close()
        if scratch.exists():
            scratch.unlink()

    print(f"\n{metadata_path}\n{candidates_path}\nkept {kept}/{len(todo)} new sentences this run")


if __name__ == "__main__":
    main()
