from __future__ import annotations

import re
from typing import Literal

Language = Literal["en", "zh"]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def detect_target_language(user_text: str) -> Language:
    text = user_text or ""
    cjk = _count(_CJK_RE, text)
    latin = _count(_LATIN_RE, text)

    # Prefer English when the user question is predominantly Latin-script,
    # even if it mentions a few CJK column names/values.
    if latin >= 8 and latin >= cjk * 2:
        return "en"
    if cjk >= 2 and cjk > latin:
        return "zh"

    # Default to English to prevent unexpected Chinese replies.
    return "en"


def language_label(language: Language) -> str:
    return "English" if language == "en" else "Chinese"


def localize(language: Language, *, en: str, zh: str) -> str:
    return en if language == "en" else zh


def is_language_mismatch(expected: Language, text: str) -> bool:
    content = text or ""
    cjk = _count(_CJK_RE, content)
    latin = _count(_LATIN_RE, content)

    # Allow some mixed-language tokens (e.g., CJK column names) without
    # treating it as a mismatch.
    if expected == "en":
        return cjk >= 20 and cjk > latin * 1.2
    return latin >= 30 and latin > cjk * 1.2


def rewrite_system_prompt(target_language: Language) -> str:
    label = language_label(target_language)
    return (
        "You are a rewriting assistant.\n"
        f"Rewrite the user's provided text so the explanation is entirely in {label}.\n"
        f"- Preserve code blocks, JSON, numbers, and any column names/values in backticks or quotes.\n"
        f"- Do not add new information.\n"
        f"- Output ONLY the rewritten text.\n"
    )
