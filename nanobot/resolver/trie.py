"""TrieIndex — Aho-Corasick trie for full-text entity scanning with incremental add."""

from __future__ import annotations

from .confidence import has_word_boundary, is_cjk, match_confidence
from .db import EntityDB


class TrieIndex:
    """Aho-Corasick trie for scanning text for known entities."""

    def __init__(self):
        self._automaton = None
        self.patterns: list[str] = []
        self.pattern_map: dict[str, list[dict]] = {}

    @property
    def loaded(self) -> bool:
        return self._automaton is not None or bool(self.pattern_map)

    def build(self, db: EntityDB, indicators: dict[str, tuple[str, list[str]]]):
        """Full rebuild from DB + indicator dict."""
        self.patterns = []
        self.pattern_map = {}

        entries = db.iter_symbols()
        alias_to_symbols = db.iter_aliases()

        # Index by symbol for alias lookup
        entry_map: dict[str, dict] = {e["symbol"]: e for e in entries}

        for symbol, info in entry_map.items():
            self._add_pattern(symbol.lower(), info, "symbol_exact")
            name_key = info["name"].lower()
            if len(name_key) >= 3:
                self._add_pattern(name_key, info, "name_exact")

        for alias_lower, symbols in alias_to_symbols.items():
            for symbol in symbols:
                if symbol in entry_map:
                    self._add_pattern(alias_lower, entry_map[symbol], "alias_exact")

        # Inject technical indicators
        for ind_key, (ind_display, ind_aliases) in indicators.items():
            ind_info = {"symbol": ind_key, "name": ind_display, "type": "technical_indicator"}
            self._add_pattern(ind_key, ind_info, "indicator_key")
            dn = ind_display.lower()
            if len(dn) >= 3 and dn != ind_key:
                self._add_pattern(dn, ind_info, "indicator_name")
            for alias in ind_aliases:
                al = alias.lower()
                if len(al) >= 3:
                    self._add_pattern(al, ind_info, "indicator_alias")

        self._rebuild_automaton()

    def add_entries(self, entries: list[dict]):
        """Incrementally add entries and rebuild automaton.

        Each entry: {symbol, name, type, exchange?, ...} or
        for indicators: {symbol (=key), name, type: "technical_indicator"}
        Also accepts optional "aliases": [str] list on each entry.
        """
        for info in entries:
            self._add_pattern(info["symbol"].lower(), info, "symbol_exact")
            name_key = info["name"].lower()
            if len(name_key) >= 3:
                mt = "indicator_name" if info.get("type") == "technical_indicator" else "name_exact"
                self._add_pattern(name_key, info, mt)
            for alias in info.get("aliases", []):
                al = alias.lower()
                if len(al) >= 3:
                    mt = "indicator_alias" if info.get("type") == "technical_indicator" else "alias_exact"
                    self._add_pattern(al, info, mt)

        self._rebuild_automaton()

    def scan(self, text: str) -> list[dict]:
        """Aho-Corasick full-text scan. Returns matches with confidence."""
        if self._automaton is None or not self.patterns:
            return []

        text_lower = text.lower()
        raw_matches = self._automaton.find_matches_as_indexes(text_lower, overlapping=True)

        match_list = []
        for pat_idx, start, end in raw_matches:
            match_list.append((pat_idx, start, end, end - start))
        match_list.sort(key=lambda x: -x[3])

        results: list[dict] = []
        used_ranges: list[tuple[int, int]] = []

        for pat_idx, start, end, length in match_list:
            matched_text = text_lower[start:end]
            if not is_cjk(matched_text) and not has_word_boundary(text_lower, start, end):
                continue

            overlaps = False
            for us, ue in used_ranges:
                if start < ue and end > us:
                    overlaps = True
                    break
            if overlaps:
                continue

            used_ranges.append((start, end))
            info_list = self.pattern_map.get(self.patterns[pat_idx], [])
            if not info_list:
                continue

            original_mention = text[start:end]
            confidence, note = match_confidence(
                matched_text, text, start, end, info_list[0].get("match_type", "unknown")
            )

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

    # ── Internal ──────────────────────────────────────

    def _add_pattern(self, key: str, info: dict, match_type: str):
        entry = {**info, "match_type": match_type}
        if key not in self.pattern_map:
            self.patterns.append(key)
            self.pattern_map[key] = [entry]
        else:
            if not any(e["symbol"] == info["symbol"] for e in self.pattern_map[key]):
                self.pattern_map[key].append(entry)

    def _rebuild_automaton(self):
        try:
            import ahocorasick_rs
            self._automaton = ahocorasick_rs.AhoCorasick(self.patterns)
        except ImportError:
            self._automaton = None
