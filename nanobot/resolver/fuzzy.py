"""FuzzyIndex — RapidFuzz matching for typos and partial names."""

from __future__ import annotations

import re

from .confidence import is_in_range


class FuzzyIndex:
    """RapidFuzz-based fuzzy matcher for entity resolution."""

    def __init__(self):
        self._choices: list[str] = []
        self._map: dict[str, dict] = {}

    @property
    def loaded(self) -> bool:
        return bool(self._choices)

    def build(self, pattern_map: dict[str, list[dict]]):
        """Build fuzzy index from trie's pattern_map."""
        self._choices = []
        self._map = {}
        for key, info_list in pattern_map.items():
            if len(key) >= 3:
                non_ind = [e for e in info_list if e.get("type") != "technical_indicator"]
                if non_ind:
                    self._choices.append(key)
                    self._map[key] = non_ind[0]

    def add_entries(self, entries: list[tuple[str, dict]]):
        """Incrementally add (key, info) pairs to fuzzy index."""
        for key, info in entries:
            if len(key) >= 3 and info.get("type") != "technical_indicator":
                if key not in self._map:
                    self._choices.append(key)
                self._map[key] = info

    def match_text(self, text: str, trie_matches: list[dict]) -> list[dict]:
        """Fallback fuzzy: find unresolved uppercase/capitalized words in text."""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return []

        if not self._choices:
            return []

        matched_ranges = {(m["start"], m["end"]) for m in trie_matches}
        candidates: list[str] = []

        for m in re.finditer(r"\b[A-Z]{2,5}\b", text):
            if not is_in_range(m.start(), m.end(), matched_ranges):
                candidates.append(m.group())

        for m in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b", text):
            if not is_in_range(m.start(), m.end(), matched_ranges):
                candidates.append(m.group())

        if not candidates:
            return []

        results: list[dict] = []
        seen_symbols: set[str] = {m["symbol"] for m in trie_matches}

        for candidate in candidates:
            best = process.extract(
                candidate.lower(), self._choices, scorer=fuzz.WRatio,
                score_cutoff=70, limit=3,
            )
            for matched_text, score, idx in best:
                info = self._map.get(matched_text)
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
                results.append({
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
                })

        return results

    def match_entities(
        self, unresolved: list[dict], seen_symbols: set[str]
    ) -> list[dict]:
        """NER path: fuzzy match for entities that didn't match dictionary exactly."""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return []

        if not self._choices or not unresolved:
            return []

        results: list[dict] = []
        for entity in unresolved:
            text = entity["text"]
            ner_conf = entity.get("confidence", 0.5)
            fuzzy_input = (entity.get("name") or text).lower().strip()

            best = process.extract(
                fuzzy_input, self._choices, scorer=fuzz.WRatio,
                score_cutoff=75, limit=3,
            )

            for matched_text, score, idx in best:
                info = self._map.get(matched_text)
                if not info or info["symbol"] in seen_symbols:
                    continue

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
