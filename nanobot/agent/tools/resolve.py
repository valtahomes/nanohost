"""Ticker resolver: GLiNER2 NER + dictionary lookup + RapidFuzz.

Primary pipeline (when NER API available):
  Stage 0 (NER):   GLiNER2 extracts financial entities from text (semantic pre-filter)
  Stage 1 (Dict):  O(1) dictionary lookup for extracted entities
  Stage 2 (Fuzzy): RapidFuzz catches typos in unresolved entities
  Stage 3 (LLM):   Candidates returned → calling LLM does semantic verification

Fallback pipeline (when NER API unavailable):
  Stage 1 (Trie):  Aho-Corasick full-text scan (high recall, more false positives)
  Stage 2 (Fuzzy): RapidFuzz for unresolved segments
  Stage 3 (LLM):   Same as above

The NER stage eliminates false positives like "cat"→CAT, "want"→WANT, "buy"→BUY
by only extracting text that GLiNER2 identifies as financial entities.
GLiNER2 runs as a cloud API (api.fastino.ai), proxied through Platform /api/v1/ner.
"""

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

# ── Config ────────────────────────────────────────

DB_PATH = Path.home() / ".nanobot" / "cache" / "symbols.db"
CACHE_TTL = 86400  # 24 hours
NER_TIMEOUT = 4  # seconds for NER API call
NER_MAX_FAILURES = 3  # consecutive failures before circuit breaker trips
NER_COOLDOWN_SECS = 300  # 5 min cooldown after circuit breaker trips

# Module-level state (loaded once, reused across calls)
_automaton = None
_dict_loaded = False  # True once pattern_map is populated (independent of Trie)
_patterns: list[str] = []
_pattern_map: dict[str, list[dict]] = {}  # key → list of entries (same name can map to multiple symbols)
_fuzzy_choices: list[str] = []
_fuzzy_map: dict[str, dict] = {}
_db_version: str | None = None  # version of the SQLite the current Trie was built from
_last_sync_check: float = 0  # last time we checked platform API for updates

# NER circuit breaker state
_ner_fail_count = 0
_ner_cooldown_until = 0.0


# ── SQLite helpers ────────────────────────────────

def _db_get_meta(db: sqlite3.Connection, key: str) -> str | None:
    try:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def _db_set_meta(db: sqlite3.Connection, key: str, value: str):
    db.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    db.commit()


# ── Dictionary sync ───────────────────────────────

def _ensure_dictionary(api_key: str | None, api_base: str | None):
    """Two-layer cache: memory automaton → local SQLite file (downloaded from R2).

    Key principle: Trie is only rebuilt when the SQLite file actually changes.
    We check the platform API periodically (every 24h) for a new version,
    but if the version hasn't changed, the Trie stays in memory as-is.

    Flow:
      1. If in-memory automaton exists → check if we need to sync (24h interval)
         a. If no sync needed → use existing Trie (zero cost)
         b. If sync check says same version → use existing Trie (zero cost)
         c. If sync check says new version → download + rebuild Trie (~0.7s)
      2. If no automaton (first call / process restart):
         a. If local SQLite exists → build Trie from it, then background sync
         b. If no local file → download from R2, then build Trie
    """
    global _automaton, _db_version, _last_sync_check

    # ── Trie already loaded ──
    if _automaton is not None:
        # Periodically check for updates (every 24h), but don't rebuild unless version changed
        if time.time() - _last_sync_check < CACHE_TTL:
            return  # Not time to check yet → use existing Trie

        # Time to check platform API for new version
        _last_sync_check = time.time()
        if api_key and api_base:
            new_version = _sync_from_platform(api_key, api_base)
            if new_version and new_version != _db_version:
                # SQLite was updated → rebuild Trie
                _rebuild_from_local_db()
        return

    # ── First call: no Trie in memory yet ──

    # Try local SQLite first (fast: ~0.7s to build Trie)
    if DB_PATH.exists():
        _rebuild_from_local_db()

    # Check platform for updates (may download new SQLite)
    if api_key and api_base:
        new_version = _sync_from_platform(api_key, api_base)
        if new_version and new_version != _db_version:
            _rebuild_from_local_db()

    _last_sync_check = time.time()


def _rebuild_from_local_db():
    """Build Trie from the local SQLite file."""
    global _automaton, _db_version
    if not DB_PATH.exists():
        return
    try:
        db = sqlite3.connect(str(DB_PATH))
        _db_version = _db_get_meta(db, "version")
        _build_automaton_from_db(db)
        db.close()
    except Exception:
        pass


def _sync_from_platform(api_key: str, api_base: str) -> str | None:
    """Check platform API for version, download SQLite from R2 if newer.

    Returns the new version string if SQLite was updated, None otherwise.
    """
    try:
        # Get local version
        local_version = _db_version
        if not local_version and DB_PATH.exists():
            try:
                db = sqlite3.connect(str(DB_PATH))
                local_version = _db_get_meta(db, "version")
                db.close()
            except Exception:
                pass

        # Check platform API for latest version
        headers = {"Authorization": f"Bearer {api_key}"}
        if local_version:
            headers["If-None-Match"] = f'"{local_version}"'

        resp = httpx.get(
            f"{api_base}/symbols/dictionary",
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 304:
            return None  # Same version, no update needed

        if resp.status_code != 200:
            return None

        data = resp.json()
        remote_version = data.get("version", "")
        download_url = data.get("url", "")

        if not download_url:
            return None

        # If versions match, skip download
        if local_version and local_version == remote_version:
            return None

        # Download SQLite file from R2
        _download_sqlite(download_url, remote_version)
        return remote_version

    except Exception:
        return None  # Network failure → use stale local data


def _download_sqlite(url: str, version: str):
    """Download SQLite file from R2 and atomically replace local copy."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp file first (atomic replace)
    fd, tmp_path = tempfile.mkstemp(suffix=".db", dir=str(DB_PATH.parent))
    try:
        os.close(fd)
        with httpx.stream("GET", url, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_bytes(8192):
                    f.write(chunk)

        # Verify it's a valid SQLite file
        db = sqlite3.connect(tmp_path)
        count = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        if count == 0:
            db.close()
            os.unlink(tmp_path)
            return

        # Write sync metadata
        _db_set_meta(db, "last_sync", str(time.time()))
        if not _db_get_meta(db, "version"):
            _db_set_meta(db, "version", version)
        db.close()

        # Atomic replace
        shutil.move(tmp_path, str(DB_PATH))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Build automaton from SQLite ───────────────────

def _build_automaton_from_db(db: sqlite3.Connection):
    """Load all symbols + aliases from SQLite → build Aho-Corasick automaton."""
    global _automaton, _dict_loaded, _patterns, _pattern_map, _fuzzy_choices, _fuzzy_map

    entries: dict[str, dict] = {}  # symbol → {symbol, name, type, exchange}

    # Load symbols (including all metadata)
    try:
        cols = [r[1] for r in db.execute("PRAGMA table_info(symbols)").fetchall()]
        has_delisted = "delisted" in cols
    except Exception:
        has_delisted = False

    query = "SELECT symbol, name, type, exchange, country, currency" + (", delisted" if has_delisted else "") + " FROM symbols"
    for row in db.execute(query):
        symbol, name, stype, exchange, country, currency = row[:6]
        delisted = bool(row[6]) if has_delisted and len(row) > 6 else False
        entries[symbol] = {
            "symbol": symbol,
            "name": name,
            "type": stype,
            "exchange": exchange,
            "country": country,
            "currency": currency,
            "delisted": delisted,
        }

    # Load aliases (one alias can map to multiple symbols)
    alias_to_symbols: dict[str, list[str]] = {}  # alias_lower → [symbol1, symbol2, ...]
    for row in db.execute("SELECT alias, symbol FROM aliases"):
        alias, symbol = row
        if symbol in entries:
            alias_to_symbols.setdefault(alias.lower(), []).append(symbol)

    # Build patterns: symbol + name + all aliases (lowercased)
    # Each pattern key maps to a LIST of entries (for disambiguation)
    _patterns = []
    _pattern_map = {}

    def _add_pattern(key: str, info: dict, match_type: str):
        entry = {**info, "match_type": match_type}
        if key not in _pattern_map:
            _patterns.append(key)
            _pattern_map[key] = [entry]
        else:
            # Avoid duplicates by symbol
            if not any(e["symbol"] == info["symbol"] for e in _pattern_map[key]):
                _pattern_map[key].append(entry)

    for symbol, info in entries.items():
        _add_pattern(symbol.lower(), info, "symbol_exact")

        # Also add the name as a pattern
        name_key = info["name"].lower()
        if len(name_key) >= 3:
            _add_pattern(name_key, info, "name_exact")

    for alias_lower, symbols in alias_to_symbols.items():
        for symbol in symbols:
            if symbol in entries:
                _add_pattern(alias_lower, entries[symbol], "alias_exact")

    _dict_loaded = True

    # Build Aho-Corasick automaton
    try:
        import ahocorasick_rs

        _automaton = ahocorasick_rs.AhoCorasick(_patterns)
    except ImportError:
        _automaton = None  # Fallback: no trie, only fuzzy

    # Build fuzzy choices (names + aliases for rapidfuzz)
    # Only include patterns >= 3 chars (fuzzy on 1-2 char strings is meaningless)
    _fuzzy_choices = []
    _fuzzy_map = {}
    for key, info_list in _pattern_map.items():
        if len(key) >= 3:
            _fuzzy_choices.append(key)
            _fuzzy_map[key] = info_list[0]  # first entry for fuzzy scoring


# ── Stage 1: Trie scan ───────────────────────────

def _has_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if match has word boundaries (not substring of larger word)."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def _is_cjk(text: str) -> bool:
    """Check if text contains CJK/Japanese/Korean characters (no word boundary needed)."""
    return any(
        '\u4e00' <= c <= '\u9fff'    # CJK Unified Ideographs
        or '\u3040' <= c <= '\u309f'  # Hiragana
        or '\u30a0' <= c <= '\u30ff'  # Katakana
        or '\uac00' <= c <= '\ud7af'  # Hangul
        for c in text
    )


def _match_confidence(mention: str, original_text: str, start: int, end: int, match_type: str) -> tuple[float, str | None]:
    """Score confidence based on match characteristics. NO stopword filtering.

    Philosophy: Algorithm does recall, LLM does precision.
    We score confidence to help LLM prioritize, but never discard candidates.

    Scoring heuristics (purely structural, no semantic judgment):
    - Multi-word matches (e.g. "Goldman Sachs") → very likely real → 1.0
    - CJK matches (e.g. "苹果", "腾讯") → very likely real → 1.0
    - Uppercase in original text (e.g. "AAPL", "BTC") → likely ticker → 0.95
    - Long matches (5+ chars) → more likely real → 0.9
    - Medium matches (3-4 chars) → could be ticker → 0.8
    - Short matches (2 chars) → ambiguous → 0.5
    - Single char → very ambiguous → 0.3
    """
    note = None

    # Multi-word or CJK → high confidence
    if ' ' in mention or _is_cjk(mention):
        return 1.0, None

    length = len(mention)

    # Single character — very ambiguous (A, V, F, T are tickers but also letters/articles)
    if length == 1:
        return 0.3, "Single-char match — verify from context whether this is a ticker"

    # Two characters — ambiguous (AT, ON, DO, IT are both words and tickers)
    if length == 2:
        return 0.5, "Short match — verify from context"

    # Check if uppercase in original text → user likely typed a ticker
    original_mention = original_text[start:end]
    if original_mention.isupper() and length >= 2:
        return 0.95, None

    # 5+ char matches → likely company/asset name
    if length >= 5:
        return 0.9, None

    # 3-4 chars — moderate confidence
    return 0.8, None


def _trie_scan(text: str) -> list[dict]:
    """Stage 1: Aho-Corasick trie scan for all known symbols/aliases.

    Returns ALL matches with confidence scores. No filtering — LLM decides.
    Priority: longer matches first (subsume shorter overlapping ones).
    """
    if _automaton is None or not _patterns:
        return []

    text_lower = text.lower()
    raw_matches = _automaton.find_matches_as_indexes(text_lower, overlapping=True)

    results: list[dict] = []
    used_ranges: list[tuple[int, int]] = []

    # Sort by match length descending (longest match first)
    match_list = []
    for pat_idx, start, end in raw_matches:
        match_list.append((pat_idx, start, end, end - start))
    match_list.sort(key=lambda x: -x[3])

    for pat_idx, start, end, length in match_list:
        # Word boundary check (Latin scripts need spaces, CJK doesn't)
        matched_text = text_lower[start:end]
        if not _is_cjk(matched_text) and not _has_word_boundary(text_lower, start, end):
            continue

        # Check overlap with already-accepted longer matches
        overlaps = False
        for us, ue in used_ranges:
            if start < ue and end > us:
                overlaps = True
                break
        if overlaps:
            continue

        used_ranges.append((start, end))

        info_list = _pattern_map.get(_patterns[pat_idx], [])
        if not info_list:
            continue

        original_mention = text[start:end]
        confidence, note = _match_confidence(
            matched_text, text, start, end,
            info_list[0].get("match_type", "unknown")
        )

        # Emit one candidate per matching symbol (for disambiguation)
        for info in info_list:
            result = {
                "mention": original_mention,
                "start": start,
                "end": end,
                "symbol": info["symbol"],
                "name": info["name"],
                "type": info.get("type", "stock"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "currency": info.get("currency"),
                "confidence": confidence,
                "match_type": info.get("match_type", "unknown"),
            }
            if info.get("delisted"):
                result["delisted"] = True
            if note:
                result["note"] = note
            results.append(result)

    return results


# ── Stage 2: Fuzzy matching ──────────────────────

def _fuzzy_resolve(text: str, trie_matches: list[dict]) -> list[dict]:
    """Stage 2: Fuzzy match for unresolved candidates (typos, partial names)."""
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        return []

    if not _fuzzy_choices:
        return []

    matched_ranges = {(m["start"], m["end"]) for m in trie_matches}
    candidates: list[str] = []

    # Uppercase words 2-5 chars (potential tickers)
    for m in re.finditer(r"\b[A-Z]{2,5}\b", text):
        if not _is_in_range(m.start(), m.end(), matched_ranges):
            candidates.append(m.group())

    # Capitalized word sequences (potential company names)
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text):
        if not _is_in_range(m.start(), m.end(), matched_ranges):
            candidates.append(m.group())

    if not candidates:
        return []

    results: list[dict] = []
    seen_symbols: set[str] = {m["symbol"] for m in trie_matches}

    for candidate in candidates:
        best = process.extract(
            candidate.lower(),
            _fuzzy_choices,
            scorer=fuzz.WRatio,
            score_cutoff=70,
            limit=3,
        )

        for matched_text, score, idx in best:
            info = _fuzzy_map.get(matched_text)
            if not info or info["symbol"] in seen_symbols:
                continue

            len_ratio = min(len(candidate), len(matched_text)) / max(
                len(candidate), len(matched_text)
            )
            if len_ratio < 0.4:
                continue

            if score >= 90:
                confidence = 0.9
            elif score >= 80:
                confidence = 0.75
            elif score >= 70:
                confidence = 0.6
            else:
                continue

            seen_symbols.add(info["symbol"])
            results.append(
                {
                    "mention": candidate,
                    "symbol": info["symbol"],
                    "name": info["name"],
                    "type": info.get("type", "stock"),
                    "exchange": info.get("exchange"),
                    "country": info.get("country"),
                    "currency": info.get("currency"),
                    "confidence": confidence,
                    "match_type": f"fuzzy_{score:.0f}",
                    "note": f"Fuzzy match (score {score:.0f}): '{candidate}' → '{matched_text}'",
                }
            )

    return results


def _is_in_range(start: int, end: int, ranges: set[tuple[int, int]]) -> bool:
    for rs, re_ in ranges:
        if start < re_ and end > rs:
            return True
    return False


# ── Context windows ───────────────────────────────

def _add_context(text: str, candidates: list[dict]) -> list[dict]:
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


# ── Merge + dedup ─────────────────────────────────

def _merge_results(trie_matches: list[dict], fuzzy_matches: list[dict]) -> list[dict]:
    """Merge trie + fuzzy results, dedup by symbol (keep highest confidence).

    When the same mention maps to multiple symbols (e.g., BTC → stock AND crypto),
    all are kept — this is important for LLM disambiguation.

    Priority sorting ensures the 15-candidate cap keeps the most likely real tickers:
    high confidence (long/multi-word/uppercase) before low confidence (short/ambiguous).
    """
    by_symbol: dict[str, dict] = {}

    for m in trie_matches + fuzzy_matches:
        sym = m["symbol"]
        if sym not in by_symbol or m["confidence"] > by_symbol[sym]["confidence"]:
            by_symbol[sym] = m

    # Sort: high confidence first, then longer mention first (tiebreaker)
    results = sorted(
        by_symbol.values(),
        key=lambda x: (-x["confidence"], -len(x.get("mention", ""))),
    )
    return results[:15]


# ── Stage 0: NER (GLiNER2 via Platform proxy) ───

def _ner_available() -> bool:
    """Check if NER API is available (circuit breaker not tripped)."""
    if _ner_fail_count >= NER_MAX_FAILURES and time.time() < _ner_cooldown_until:
        return False
    return True


async def _ner_extract(text: str, api_key: str, api_base: str) -> list[dict] | None:
    """Call Platform NER endpoint (GLiNER2 proxy) to extract financial entities.

    Returns list of {"text": "Apple", "type": "company name", "confidence": 0.92,
    "start": 14, "end": 19} or None on failure (triggers Aho-Corasick fallback).

    Circuit breaker: after NER_MAX_FAILURES consecutive failures, skip NER for
    NER_COOLDOWN_SECS and go straight to Aho-Corasick fallback.
    """
    global _ner_fail_count, _ner_cooldown_until

    if not _ner_available():
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_base}/ner",
                json={"text": text},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=NER_TIMEOUT,
            )

        if resp.status_code != 200:
            _ner_fail_count += 1
            if _ner_fail_count >= NER_MAX_FAILURES:
                _ner_cooldown_until = time.time() + NER_COOLDOWN_SECS
            return None

        # Success — reset circuit breaker
        _ner_fail_count = 0

        data = resp.json()
        entities = data.get("entities", [])

        # Gemini format: [{"mention", "name", "ticker", "type"}]
        result: list[dict] = []
        if isinstance(entities, list):
            for item in entities:
                if isinstance(item, dict):
                    result.append({
                        "text": item.get("mention", ""),
                        "name": item.get("name"),
                        "ticker": item.get("ticker"),
                        "type": item.get("type", "company"),
                        "confidence": 0.9,
                    })
        return result

    except Exception:
        _ner_fail_count += 1
        if _ner_fail_count >= NER_MAX_FAILURES:
            _ner_cooldown_until = time.time() + NER_COOLDOWN_SECS
        return None  # Fallback to Aho-Corasick


# ── NER Stage 1: Dictionary lookup ───────────────

def _dict_lookup(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """O(1) dictionary lookup for NER-extracted entities.

    Gemini provides ticker/name/mention — try all three for higher hit rate.
    Returns (resolved, unresolved).
    """
    resolved: list[dict] = []
    unresolved: list[dict] = []
    seen_symbols: set[str] = set()

    for entity in entities:
        mention = entity["text"]
        ner_conf = entity.get("confidence", 0.5)

        # Try matching in order: ticker → name → mention (3 chances)
        candidates = [entity.get("ticker"), entity.get("name"), mention]
        info_list = None
        matched_key = None
        for candidate in candidates:
            if not candidate:
                continue
            key = candidate.lower().strip()
            found = _pattern_map.get(key, [])
            if found:
                info_list = found
                matched_key = key
                break

        if info_list:
            for info in info_list:
                if info["symbol"] in seen_symbols:
                    continue
                seen_symbols.add(info["symbol"])

                # Dictionary match quality score
                mt = info.get("match_type", "")
                if "symbol" in mt:
                    dict_score = 1.0
                elif "name" in mt:
                    dict_score = 0.95
                else:
                    dict_score = 0.90  # alias

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


# ── NER Stage 2: Fuzzy for unresolved entities ───

def _fuzzy_resolve_entities(
    unresolved: list[dict], seen_symbols: set[str]
) -> list[dict]:
    """RapidFuzz for NER entities that didn't match dictionary exactly.

    Higher score_cutoff (75) than fallback fuzzy (70) because NER already
    confirmed these are financial entities — less need for aggressive matching.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        return []

    if not _fuzzy_choices or not unresolved:
        return []

    results: list[dict] = []
    for entity in unresolved:
        text = entity["text"]
        ner_conf = entity.get("confidence", 0.5)
        # Prefer English name for fuzzy matching (higher hit rate than CJK mention)
        fuzzy_input = (entity.get("name") or text).lower().strip()

        best = process.extract(
            fuzzy_input,
            _fuzzy_choices,
            scorer=fuzz.WRatio,
            score_cutoff=75,
            limit=3,
        )

        for matched_text, score, idx in best:
            info = _fuzzy_map.get(matched_text)
            if not info or info["symbol"] in seen_symbols:
                continue

            # Length ratio filter — prevent short→long mismatches
            len_ratio = min(len(fuzzy_input), len(matched_text)) / max(
                len(fuzzy_input), len(matched_text)
            )
            if len_ratio < 0.4:
                continue

            dict_score = score / 100 * 0.85
            confidence = round(0.35 * ner_conf + 0.65 * dict_score, 3)

            seen_symbols.add(info["symbol"])
            results.append({
                "mention": text,
                "symbol": info["symbol"],
                "name": info["name"],
                "type": info.get("type", "stock"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "currency": info.get("currency"),
                "confidence": confidence,
                "match_type": f"ner_fuzzy_{score:.0f}",
                "note": f"Fuzzy match (score {score:.0f}): '{text}' → '{matched_text}'",
            })

    return results


# ── Tool class ────────────────────────────────────

class TickerResolverTool(Tool):
    """Resolve company/asset names to FMP ticker symbols."""

    name = "resolve_tickers"
    description = (
        "Resolve company names, asset names, abbreviations, and ambiguous text "
        "to exact ticker symbols. Call this BEFORE stock_lookup, technical_indicators, "
        "or any financial data tool when the user mentions names instead of tickers. "
        "Returns candidates with confidence scores for verification."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "User's text containing company/asset names to resolve",
            }
        },
        "required": ["text"],
    }

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    async def execute(self, text: str, **kwargs: Any) -> str:
        _ensure_dictionary(self.api_key, self.api_base)

        # ── Primary pipeline: NER → Dict → Fuzzy ──
        ner_entities = None
        if self.api_key and self.api_base:
            ner_entities = await _ner_extract(text, self.api_key, self.api_base)

        if ner_entities is not None:
            # NER succeeded (may be empty = no financial entities in text)
            if not ner_entities:
                return json.dumps(
                    {"candidates": [], "note": "No financial entities detected in text"}
                )

            # Stage 1: Dictionary lookup (O(1) per entity)
            resolved, unresolved = _dict_lookup(ner_entities)

            # Stage 2: Fuzzy for unresolved entities
            seen = {r["symbol"] for r in resolved}
            fuzzy = _fuzzy_resolve_entities(unresolved, seen)

            all_candidates = resolved + fuzzy
            # Cap at 15 candidates (sorted by confidence), same as fallback path
            all_candidates.sort(key=lambda x: (-x["confidence"], -len(x.get("mention", ""))))
            all_candidates = all_candidates[:15]
        else:
            # ── Fallback: Aho-Corasick full-text scan ──
            trie_matches = _trie_scan(text)
            fuzzy_matches = _fuzzy_resolve(text, trie_matches)
            all_candidates = _merge_results(trie_matches, fuzzy_matches)

        # Add context windows for LLM verification
        all_candidates = _add_context(text, all_candidates)

        if not all_candidates:
            return json.dumps(
                {"candidates": [], "note": "No financial entities detected in text"}
            )

        return json.dumps(
            {
                "instruction": (
                    "Below are ticker candidates extracted from the user's message. "
                    "Review each candidate's context to determine if it refers to a financial asset. "
                    "Discard false positives (common words mistaken for tickers). "
                    "Use confirmed symbols for subsequent data API calls. "
                    "If unsure about a candidate, ask the user."
                ),
                "candidates": all_candidates,
            }
        )
