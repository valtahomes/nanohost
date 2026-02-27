"""Resolver — pipeline orchestrator with incremental expansion API."""

from __future__ import annotations

import json
from pathlib import Path

from .confidence import add_context, dict_lookup, merge_results
from .db import DB_PATH, EntityDB
from .fuzzy import FuzzyIndex
from .indicators import INDICATOR_NAMES
from .ner import NERClient
from .trie import TrieIndex


class Resolver:
    """Financial entity resolver: NER → Dict → Fuzzy pipeline.

    Supports incremental expansion and direct programmatic use.

    Usage:
        r = Resolver(api_key="...", api_base="https://api.teleclaws.com/v1")
        result = await r.resolve("AAPL RSI超过70时提醒我")

        # Incremental expansion
        r.add_symbols([{"symbol": "NEWCO", "name": "New Co", "type": "stock"}])
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        db_path: Path = DB_PATH,
    ):
        self.db = EntityDB(db_path)
        self.trie = TrieIndex()
        self.fuzzy = FuzzyIndex()
        self.ner = NERClient(api_key, api_base) if api_key and api_base else None
        self._api_key = api_key
        self._api_base = api_base
        self._loaded = False

    def ensure_loaded(self):
        """Lazy load: DB → trie → fuzzy. Syncs with platform if needed."""
        if self._loaded and self.trie.loaded:
            # Check for platform updates periodically
            if self._api_key and self._api_base:
                updated = self.db.sync_from_platform(self._api_key, self._api_base)
                if updated:
                    self._build_indexes()
            return

        # First load
        if self.db.path.exists():
            self._build_indexes()

        if self._api_key and self._api_base:
            updated = self.db.sync_from_platform(self._api_key, self._api_base)
            if updated:
                self._build_indexes()

        self._loaded = True

    async def resolve(self, text: str) -> str:
        """Full resolution pipeline. Returns JSON string with candidates."""
        self.ensure_loaded()

        # ── Primary pipeline: NER → Dict → Fuzzy ──
        ner_entities = None
        if self.ner and self.ner.available():
            ner_entities = await self.ner.extract(text)

        if ner_entities is not None:
            if not ner_entities:
                return json.dumps(
                    {"candidates": [], "note": "No financial entities detected in text"}
                )

            resolved, unresolved = dict_lookup(ner_entities, self.trie.pattern_map)
            seen = {r["symbol"] for r in resolved}
            fuzzy = self.fuzzy.match_entities(unresolved, seen)

            all_candidates = resolved + fuzzy
            all_candidates.sort(
                key=lambda x: (-x["confidence"], -len(x.get("mention", "")))
            )
            all_candidates = all_candidates[:15]
        else:
            # ── Fallback: Aho-Corasick full-text scan ──
            trie_matches = self.trie.scan(text)
            fuzzy_matches = self.fuzzy.match_text(text, trie_matches)
            all_candidates = merge_results(trie_matches, fuzzy_matches)

        all_candidates = add_context(text, all_candidates)

        if not all_candidates:
            return json.dumps(
                {"candidates": [], "note": "No financial entities detected in text"}
            )

        return json.dumps({
            "instruction": (
                "Below are ticker and indicator candidates extracted from the user's message. "
                "Review each candidate's context to determine if it refers to a financial asset or indicator. "
                "Discard false positives (common words mistaken for tickers). "
                "Use confirmed symbols for subsequent data API calls. "
                "Use confirmed indicators (type=technical_indicator) for alert_check technical conditions. "
                "If unsure about a candidate, ask the user."
            ),
            "candidates": all_candidates,
        })

    # ── Incremental API ───────────────────────────────

    def add_symbols(self, symbols: list[dict]):
        """Add symbols to DB + trie + fuzzy. Each: {symbol, name, type, ...}."""
        self.ensure_loaded()
        self.db.add_symbols(symbols)
        # Build trie entries with optional aliases
        self.trie.add_entries(symbols)
        # Update fuzzy index
        for s in symbols:
            key = s["symbol"].lower()
            self.fuzzy.add_entries([(key, s)])
            name_key = s["name"].lower()
            if len(name_key) >= 3:
                self.fuzzy.add_entries([(name_key, s)])

    def add_aliases(self, aliases: list[tuple[str, str]]):
        """Add (alias, symbol) pairs to DB + trie + fuzzy."""
        self.ensure_loaded()
        self.db.add_aliases(aliases)
        # Look up symbol info for each alias to add to trie
        for alias, symbol in aliases:
            info = self.db.get_symbol(symbol)
            if info:
                entry = {**info, "aliases": [alias]}
                self.trie.add_entries([entry])
                al = alias.lower()
                if len(al) >= 3:
                    self.fuzzy.add_entries([(al, info)])

    def add_indicators(self, indicators: dict[str, tuple[str, list[str]]]):
        """Add indicator entries to trie only (not persisted to SQLite)."""
        self.ensure_loaded()
        entries = []
        for key, (display, aliases) in indicators.items():
            entries.append({
                "symbol": key,
                "name": display,
                "type": "technical_indicator",
                "aliases": aliases,
            })
        self.trie.add_entries(entries)

    def rebuild(self):
        """Force full rebuild from DB."""
        self._build_indexes()
        self._loaded = True

    # ── Internal ──────────────────────────────────────

    def _build_indexes(self):
        self.trie.build(self.db, INDICATOR_NAMES)
        self.fuzzy.build(self.trie.pattern_map)
