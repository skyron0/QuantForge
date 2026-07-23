from typing import Dict, Optional
from backend.market_data.exceptions import NormalizationError


class MarketDataNormalizer:
    """
    Handles all symbol, side, timeframe, and payload formatting to QuantForge standards.
    Centralized mapping prevents scattering provider-specific translation rules.
    """

    def __init__(self, symbol_mappings: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        # symbol_mappings is structured as provider -> {raw_symbol: canonical_symbol}
        # e.g., {"bybit": {"BTC/USDT": "BTCUSDT"}}
        self._mappings: Dict[str, Dict[str, str]] = {}
        if symbol_mappings:
            for provider, mappings in symbol_mappings.items():
                self._mappings[provider.lower()] = {k.upper(): v.upper() for k, v in mappings.items()}

    def register_mapping(self, provider: str, raw_symbol: str, canonical_symbol: str) -> None:
        prov_key = provider.lower()
        if prov_key not in self._mappings:
            self._mappings[prov_key] = {}
        self._mappings[prov_key][raw_symbol.upper()] = canonical_symbol.upper()

    def normalize_symbol(self, provider: str, raw_symbol: str) -> str:
        if not raw_symbol or not raw_symbol.strip():
            raise NormalizationError("Raw symbol cannot be empty")

        prov_key = provider.lower()
        cleaned_raw = raw_symbol.strip().upper()

        # Check registered mapping
        if prov_key in self._mappings and cleaned_raw in self._mappings[prov_key]:
            return self._mappings[prov_key][cleaned_raw]

        # General normalizer rules: delete separators (-, _, /, @)
        cleaned = cleaned_raw.replace("/", "").replace("-", "").replace("_", "").replace("@", "")
        if not cleaned:
            raise NormalizationError(f"Symbol normalization resulted in empty string: {raw_symbol}")
        return cleaned

    def normalize_side(self, raw_side: str) -> str:
        if not raw_side:
            raise NormalizationError("Raw side cannot be empty")
        side_lower = raw_side.strip().lower()
        if side_lower in ("buy", "long", "b", "bid"):
            return "buy"
        if side_lower in ("sell", "short", "s", "ask"):
            return "sell"
        raise NormalizationError(f"Unsupported side: {raw_side}")

    def normalize_timeframe(self, raw_timeframe: str) -> str:
        if not raw_timeframe:
            raise NormalizationError("Raw timeframe cannot be empty")
        timeframe_clean = raw_timeframe.strip().lower()
        # Handle common variants
        if timeframe_clean in ("1m", "1min", "m1"):
            return "1m"
        if timeframe_clean in ("5m", "5min", "m5"):
            return "5m"
        if timeframe_clean in ("15m", "15min", "m15"):
            return "15m"
        if timeframe_clean in ("1h", "1hour", "h1"):
            return "1h"
        if timeframe_clean in ("1d", "daily", "d1"):
            return "1d"
        return timeframe_clean
