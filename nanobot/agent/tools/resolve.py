"""Ticker + indicator resolver tool — thin wrapper around nanobot.resolver.

See nanobot/resolver/ for the standalone package with incremental expansion API.
"""

import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.resolver import Resolver


class TickerResolverTool(Tool):
    """Resolve company/asset names to FMP ticker symbols."""

    name = "resolve_tickers"
    description = (
        "Resolve company names, asset names, abbreviations, technical indicator names, "
        "and ambiguous text to exact ticker symbols or indicator keys. "
        "Call this BEFORE stock_lookup, alert_check, or any financial data tool "
        "when the user mentions names instead of tickers/indicator keys. "
        "Returns candidates with confidence scores — type 'technical_indicator' "
        "for indicators (e.g. 'Chaikin oscillator' → adosc), others for tickers."
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
        self._resolver = Resolver(api_key=api_key, api_base=api_base)

    async def execute(self, text: str, **kwargs: Any) -> str:
        return await self._resolver.resolve(text)
