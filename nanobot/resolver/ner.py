"""NERClient — platform NER endpoint with circuit breaker."""

from __future__ import annotations

import time

import httpx


class NERClient:
    """NER client with circuit breaker for the platform /api/v1/ner endpoint."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        timeout: int = 4,
        max_failures: int = 3,
        cooldown_secs: int = 300,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout
        self.max_failures = max_failures
        self.cooldown_secs = cooldown_secs
        self._fail_count = 0
        self._cooldown_until = 0.0

    def available(self) -> bool:
        if self._fail_count >= self.max_failures and time.time() < self._cooldown_until:
            return False
        return True

    async def extract(self, text: str) -> list[dict] | None:
        """Call NER endpoint. Returns entities or None (triggers fallback)."""
        if not self.available():
            return None

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.api_base}/ner",
                    json={"text": text},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )

            if resp.status_code != 200:
                self._fail_count += 1
                if self._fail_count >= self.max_failures:
                    self._cooldown_until = time.time() + self.cooldown_secs
                return None

            self._fail_count = 0

            data = resp.json()
            entities = data.get("entities", [])

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
            self._fail_count += 1
            if self._fail_count >= self.max_failures:
                self._cooldown_until = time.time() + self.cooldown_secs
            return None
