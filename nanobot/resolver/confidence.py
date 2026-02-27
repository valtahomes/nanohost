"""Scoring, context windows, merge/dedup, and NER dict lookup utilities."""

from __future__ import annotations

import re
from typing import Any


def is_cjk(text: str) -> bool:
    """Check if text contains CJK/Japanese/Korean characters."""
    return any(
        '\u4e00' <= c <= '\u9fff'
        or '\u3040' <= c <= '\u309f'
        or '\u30a0' <= c <= '\u30ff'
        or '\uac00' <= c <= '\ud7af'
        for c in text
    )


def has_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if match has word boundaries (not substring of larger word)."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def match_confidence(
    mention: str, original_text: str, start: int, end: int, match_type: str
) -> tuple[float, str | None]:
    """Score confidence based on match characteristics.

    Philosophy: Algorithm does recall, LLM does precision.
    """
    note = None

    if ' ' in mention or is_cjk(mention):
        return 1.0, None

    length = len(mention)

    if length == 1:
        return 0.3, "Single-char match — verify from context whether this is a ticker"

    if length == 2:
        return 0.5, "Short match — verify from context"

    original_mention = original_text[start:end]
    if original_mention.isupper() and length >= 2:
        return 0.95, None

    if length >= 5:
        return 0.9, None

    return 0.8, None


def add_context(text: str, candidates: list[dict]) -> list[dict]:
    """Add ±30 char context window around each match for LLM verification."""
    for c in candidates:
        start = c.get("start")
        end = c.get("end")
        if start is not None and end is not None:
            ctx_start = max(0, start - 30)
            ctx_end = min(len(text), end + 30)
            before = text[ctx_start:start]
            mention = text[start:end]
            after = text[end:ctx_end]
            c["context"] = f"{before}>>>{mention}<<<{after}"
            c.pop("start", None)
            c.pop("end", None)
        elif "mention" in c:
            idx = text.lower().find(c["mention"].lower())
            if idx >= 0:
                ctx_start = max(0, idx - 30)
                ctx_end = min(len(text), idx + len(c["mention"]) + 30)
                before = text[ctx_start:idx]
                mention = text[idx : idx + len(c["mention"])]
                after = text[idx + len(c["mention"]) : ctx_end]
                c["context"] = f"{before}>>>{mention}<<<{after}"
    return candidates


def merge_results(
    trie_matches: list[dict], fuzzy_matches: list[dict], limit: int = 15
) -> list[dict]:
    """Merge trie + fuzzy results, dedup by symbol (keep highest confidence)."""
    by_symbol: dict[str, dict] = {}
    for m in trie_matches + fuzzy_matches:
        sym = m["symbol"]
        if sym not in by_symbol or m["confidence"] > by_symbol[sym]["confidence"]:
            by_symbol[sym] = m

    results = sorted(
        by_symbol.values(),
        key=lambda x: (-x["confidence"], -len(x.get("mention", ""))),
    )
    return results[:limit]


def is_in_range(start: int, end: int, ranges: set[tuple[int, int]]) -> bool:
    for rs, re_ in ranges:
        if start < re_ and end > rs:
            return True
    return False


def dict_lookup(
    entities: list[dict], pattern_map: dict[str, list[dict]]
) -> tuple[list[dict], list[dict]]:
    """O(1) dictionary lookup for NER-extracted entities.

    Returns (resolved, unresolved).
    """
    resolved: list[dict] = []
    unresolved: list[dict] = []
    seen_symbols: set[str] = set()

    for entity in entities:
        mention = entity["text"]
        ner_conf = entity.get("confidence", 0.5)

        candidates = [entity.get("ticker"), entity.get("name"), mention]
        info_list = None
        for candidate in candidates:
            if not candidate:
                continue
            key = candidate.lower().strip()
            found = pattern_map.get(key, [])
            if found:
                info_list = found
                break

        if info_list:
            for info in info_list:
                if info["symbol"] in seen_symbols:
                    continue
                seen_symbols.add(info["symbol"])

                mt = info.get("match_type", "")
                if "symbol" in mt:
                    dict_score = 1.0
                elif "name" in mt:
                    dict_score = 0.95
                else:
                    dict_score = 0.90

                confidence = round(0.35 * ner_conf + 0.65 * dict_score, 3)

                entry: dict[str, Any] = {
                    "mention": mention,
                    "symbol": info["symbol"],
                    "name": info["name"],
                    "type": info.get("type", "stock"),
                    "exchange": info.get("exchange"),
                    "country": info.get("country"),
                    "currency": info.get("currency"),
                    "confidence": confidence,
                    "match_type": f"ner_{mt}",
                }
                if info.get("delisted"):
                    entry["delisted"] = True
                resolved.append(entry)
        else:
            unresolved.append(entity)

    return resolved, unresolved
