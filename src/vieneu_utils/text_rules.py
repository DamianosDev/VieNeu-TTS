"""Rewrite Vietnamese/English code-switched text before sea-g2p normalizes it.

sea-g2p reads a compiled Vietnamese dictionary with no custom-lexicon hook, but it
passes ``<en>...</en>`` through untouched and phonemizes its content in English. This
module is the lever: :func:`apply_text_rules` runs three passes, in order, and hands
the result to the existing normalizer unchanged otherwise.

1. Overrides: a "term = spoken form" pronunciation file, longest term first, one pass.
2. Technical tokens: URLs, emails, handles, hashtags, paths, and identifiers rewritten
   into a spoken form, with English-looking segments wrapped in ``<en>``.
3. Code-switch disambiguation: a curated list of Vietnamese-syllable-shaped English
   words (``no``, ``can``, ``do``, ...) is wrapped in ``<en>`` only when the same
   clause also contains an unambiguous English word.

Existing ``<en>...</en>`` spans, ``[emotion cue]`` brackets, and ``<|emotion_k|>``
tokens are never touched, and the whole function is idempotent.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Vietnamese syllable validator (onset + nucleus + coda), used to tell an
# unambiguous English word from a Vietnamese-shaped one during code-switch
# disambiguation. Not a dictionary — a structural approximation of Vietnamese
# phonotactics, which is enough to reject English consonant clusters VN does
# not have (fr, pl, sh, st, w-, bare f/j/z, doubled vowels, ...).
# ---------------------------------------------------------------------------
_VOWELS = "aăâeêioôơuưy"
_ONSETS = ("ngh", "nh", "ng", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
           "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x")
_CODAS = ("ch", "ng", "nh", "c", "m", "n", "p", "t")
_TONE_MARKS = "̣̀́̃̉"  # huyền, sắc, ngã, hỏi, nặng


def _strip_tone(word: str) -> str:
    """Lowercase and drop tone-mark combining characters (keep ă/â/ê/ô/ơ/ư/đ)."""
    decomposed = unicodedata.normalize("NFD", word.lower())
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if ch not in _TONE_MARKS))


def _is_valid_nucleus(s: str) -> bool:
    if not (1 <= len(s) <= 3):
        return False
    if any(ch not in _VOWELS for ch in s):
        return False
    return all(s[i] != s[i + 1] for i in range(len(s) - 1))


def is_vietnamese_syllable(word: str) -> bool:
    """Whether ``word`` could be a single Vietnamese syllable (onset+nucleus+coda)."""
    if not word or not word.isalpha():
        return False
    w = _strip_tone(word)
    for onset in ("",) + _ONSETS:
        if not w.startswith(onset):
            continue
        rest = w[len(onset):]
        for coda in ("",) + _CODAS:
            nucleus = rest[:-len(coda)] if coda else rest
            if coda and not rest.endswith(coda):
                continue
            if _is_valid_nucleus(nucleus):
                return True
    return False


# ---------------------------------------------------------------------------
# Protected spans: existing <en>...</en>, [bracket cues], and <|emotion_k|>
# tokens are opaque to every pass below, which also makes apply_text_rules
# idempotent (a second call sees its own output as already-protected).
# ---------------------------------------------------------------------------
_PROTECTED_RE = re.compile(r"<en>.*?</en>|<\|[^|]*\|>|\[[^\[\]]*\]", re.DOTALL | re.IGNORECASE)


def _map_unprotected(text: str, fn) -> str:
    parts = _PROTECTED_RE.split(text)
    protected = _PROTECTED_RE.findall(text)
    out: List[str] = []
    for i, part in enumerate(parts):
        out.append(fn(part) if part else part)
        if i < len(protected):
            out.append(protected[i])
    return "".join(out)


# ---------------------------------------------------------------------------
# Pass 1: overrides ("term = spoken form"), longest term first.
# ---------------------------------------------------------------------------
def load_pronunciation_file(path: Path) -> "Dict[str, str]":
    """Parse a ``term = spoken form`` file into ``{term: spoken form}``.

    Blank lines and lines starting with ``#`` are skipped. Every other line must
    contain ``=``; invalid lines are collected and raised together, once, with
    their line numbers.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pronunciation file not found: {path}")
    overrides: Dict[str, str] = {}
    bad: List[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            bad.append(f"  line {lineno}: {raw!r}")
            continue
        term, spoken = line.split("=", 1)
        term, spoken = term.strip(), spoken.strip()
        if not term or not spoken:
            bad.append(f"  line {lineno}: {raw!r}")
            continue
        overrides[term] = spoken
    if bad:
        raise ValueError(f"Invalid pronunciation line(s) in {path}:\n" + "\n".join(bad))
    return overrides


_DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "vieneu" / "assets" / "pronunciations_default.txt"
_default_overrides_cache: Optional[Dict[str, str]] = None


def _default_overrides() -> Dict[str, str]:
    global _default_overrides_cache
    if _default_overrides_cache is None:
        _default_overrides_cache = (
            load_pronunciation_file(_DEFAULT_OVERRIDES_PATH) if _DEFAULT_OVERRIDES_PATH.is_file() else {}
        )
    return _default_overrides_cache


def _override_pattern(term: str) -> re.Pattern:
    flags = 0 if any(ch.isupper() for ch in term) else re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags)


def _apply_overrides(text: str, overrides: Dict[str, str]) -> str:
    if not overrides:
        return text
    for term in sorted(overrides, key=len, reverse=True):
        text = _map_unprotected(text, lambda part, t=term, s=overrides[term]: _override_pattern(t).sub(
            lambda m: s, part
        ))
    return text


# ---------------------------------------------------------------------------
# Pass 2: technical tokens (URLs, emails, handles, hashtags, paths, identifiers).
# ---------------------------------------------------------------------------
_SYMBOL_WORDS = {
    ".": "chấm", "/": "gạch chéo", "\\": "gạch chéo ngược",
    "_": "gạch dưới", "-": "gạch ngang", "@": "a còng",
    "?": "hỏi", "=": "bằng",
}
_SYMBOL_SPLIT_RE = re.compile(r"([./\\_@?=-])")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_TLDS = "com|net|org|io|vn|edu|gov|co|dev|info|me|app|ai"
# A path/identifier segment: alnum/underscore/dash, with optional internal
# ".ext"-style groups — never ends on a bare "." (so a sentence-final period
# is never swallowed into the match).
_SEG = r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*"
_TOKEN_RE = re.compile(
    r"(?P<url>(?:https?://)?(?:www\.)?[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.(?:%(tlds)s)(?:/[^\s,.;:!?\"']*)?"
    # An optional "?key=value" query string. Requires "key=" right after the "?" so a
    # sentence's own trailing "?" (no query string) is never absorbed into the URL.
    r"(?:\?[A-Za-z0-9_\-]+=[^\s,.;:!\"']*)?"
    r"|(?:https?://)[^\s,.;:!?\"']+)"
    r"|(?P<email>[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+)"
    r"|(?P<handle>@[A-Za-z0-9_]+)"
    r"|(?P<hashtag>\#[A-Za-z0-9_]+)"
    r"|(?P<winpath>[A-Za-z]:\\%(seg)s(?:\\%(seg)s)*)"
    r"|(?P<path>(?:/%(seg)s)+|\b%(seg)s(?:/%(seg)s)+\b)"
    r"|(?P<snake>\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b)"
    r"|(?P<dotted>\b[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)+\b)"
    r"|(?P<camel>\b[A-Za-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b)"
    r"|(?P<allcaps>\b[A-Z]{2,}\b)"
    % {"tlds": _TLDS, "seg": _SEG}
)


def _wrap_word(word: str) -> str:
    """A plain alpha chunk: ALL-CAPS (2+) is spelled letter by letter in English,
    anything else is wrapped whole (one English "word")."""
    if word.isupper() and len(word) >= 2:
        return f"<en>{' '.join(word)}</en>"
    return f"<en>{word}</en>"


def _split_camel(word: str) -> List[str]:
    return [w for w in _CAMEL_SPLIT_RE.sub(" ", word).split(" ") if w]


def _spoken_symbols(s: str) -> str:
    """Split ``s`` on ``. / \\ _ @ ? = -`` and read each chunk: symbols as their
    Vietnamese word, alpha chunks (camelCase-split) wrapped in ``<en>``, anything
    else (digits, mixed) left as-is."""
    out: List[str] = []
    for part in _SYMBOL_SPLIT_RE.split(s):
        if not part:
            continue
        if part in _SYMBOL_WORDS:
            out.append(_SYMBOL_WORDS[part])
        elif part.lower() == "www":
            out.append("<en>w w w</en>")
        elif part.isalpha():
            out.append(" ".join(_wrap_word(w) for w in _split_camel(part)))
        else:
            out.append(part)
    return " ".join(out)


def _rewrite_token(m: re.Match) -> str:
    kind = m.lastgroup
    text = m.group()
    if kind == "url":
        text = re.sub(r"^https?://", "", text)
        return _spoken_symbols(text)
    if kind == "email":
        return _spoken_symbols(text)
    if kind == "handle":
        return "a còng " + _spoken_symbols(text[1:])
    if kind == "hashtag":
        return "thăng " + _spoken_symbols(text[1:])
    if kind == "winpath":
        drive, rest = text.split(":", 1)
        rest = rest[1:] if rest.startswith("\\") else rest
        return f"<en>{drive}</en> hai chấm gạch chéo ngược " + _spoken_symbols(rest)
    if kind in ("path", "dotted"):
        if not any(ch.isalpha() for ch in text):
            return text  # a date like 24/09/2026 or 1.500.000 — leave it to the number normalizer
        return _spoken_symbols(text)
    if kind == "snake":
        return _spoken_symbols(text)
    if kind in ("camel", "allcaps"):
        return " ".join(_wrap_word(w) for w in _split_camel(text))
    return text  # pragma: no cover - unreachable, every branch above covers a group


def _apply_technical_tokens(text: str) -> str:
    return _map_unprotected(text, lambda part: _TOKEN_RE.sub(_rewrite_token, part))


# ---------------------------------------------------------------------------
# Pass 3: code-switch disambiguation.
# ---------------------------------------------------------------------------
AMBIGUOUS_WORDS = frozenset(
    "no can do it in to me so go an on a i is at be he we my by up us "
    "tin fan ban bus pin top hot set test game mode show ship".split()
)

_CLAUSE_SPLIT_RE = re.compile(r"([,.;:!?]+)")
_TOKEN_SPLIT_RE = re.compile(r"(\s+)")


def _classify(word: str) -> str:
    if not word.isalpha():
        return "OTHER"
    low = word.lower()
    if low in AMBIGUOUS_WORDS:
        return "AMB"
    return "VN" if is_vietnamese_syllable(word) else "ANCHOR"


def _code_switch_clause(clause: str) -> str:
    parts = _TOKEN_SPLIT_RE.split(clause)
    if not parts or not any(p.strip() for p in parts):
        return clause
    classes = [_classify(p) if p.strip() else "SPACE" for p in parts]
    if "ANCHOR" not in classes:
        return clause
    wrap = [c in ("AMB", "ANCHOR") for c in classes]
    changed = True
    while changed:
        changed = False
        for i in range(1, len(wrap) - 1):
            if not wrap[i] and wrap[i - 1] and wrap[i + 1]:
                wrap[i] = True
                changed = True
    out: List[str] = []
    i = 0
    n = len(parts)
    while i < n:
        if wrap[i]:
            j = i
            while j < n and wrap[j]:
                j += 1
            out.append(f"<en>{''.join(parts[i:j])}</en>")
            i = j
        else:
            out.append(parts[i])
            i += 1
    return "".join(out)


def _apply_code_switch(text: str) -> str:
    def rewrite(part: str) -> str:
        pieces = _CLAUSE_SPLIT_RE.split(part)
        return "".join(_code_switch_clause(p) if i % 2 == 0 else p for i, p in enumerate(pieces))

    return _map_unprotected(text, rewrite)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def apply_text_rules(text: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """Rewrite ``text`` before it reaches sea-g2p: overrides, then technical
    tokens, then code-switch disambiguation. Idempotent; never touches content
    already inside ``<en>``, ``[cue]``, or ``<|emotion_k|>``."""
    if not text:
        return text
    merged = dict(_default_overrides())
    if overrides:
        merged.update(overrides)
    text = _apply_overrides(text, merged)
    text = _apply_technical_tokens(text)
    text = _apply_code_switch(text)
    return text
