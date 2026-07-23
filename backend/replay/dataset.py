import hashlib
import json
from decimal import Decimal
from typing import List, Dict, Any
from backend.replay.exceptions import ReplayValidationError

def generate_dataset_hash(records: List[Dict[str, Any]]) -> str:
    """
    Generates a deterministic SHA-256 fingerprint hash for a list of market data records.
    Constructs a canonical JSON representation of stable fields.
    """
    cleaned = []
    for r in records:
        cleaned.append({
            "timestamp": str(r.get("timestamp") or r.get("open_time")),
            "symbol": str(r.get("symbol", "")).upper().replace("/", ""),
            "open": str(r.get("open", "0")),
            "high": str(r.get("high", "0")),
            "low": str(r.get("low", "0")),
            "close": str(r.get("close", "0")),
            "volume": str(r.get("volume", "0")),
            "sequence": int(r.get("sequence", 0))
        })
    # Canonical JSON string
    serialized = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def sort_records_deterministically(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts a raw dataset using an explicit stable tie-break rule:
    1. Timestamp / open_time (ascending)
    2. Symbol name (alphabetical)
    3. Sequence number if available (ascending)
    Does not mutate the input list.
    """
    def key_fn(r: Dict[str, Any]):
        ts = str(r.get("timestamp") or r.get("open_time") or "")
        sym = str(r.get("symbol") or "").upper().replace("/", "")
        seq = int(r.get("sequence") or 0)
        return (ts, sym, seq)
    return sorted(records, key=key_fn)

def validate_ohlc_record(r: Dict[str, Any]) -> None:
    """
    Performs boundary checks on a candle/OHLC record:
    - High >= Open, High >= Close, High >= Low.
    - Low <= Open, Low <= Close.
    - Prices must be positive and finite.
    """
    symbol = r.get("symbol", "UNKNOWN")
    ts = r.get("timestamp") or r.get("open_time") or "UNKNOWN"

    try:
        o = Decimal(str(r["open"]))
        h = Decimal(str(r["high"]))
        l = Decimal(str(r["low"]))
        c = Decimal(str(r["close"]))
        v = Decimal(str(r.get("volume", 0)))
    except (KeyError, ValueError, TypeError) as e:
        raise ReplayValidationError(
            f"Missing or invalid numeric field in record for {symbol} at {ts}: {str(e)}"
        )

    # Positive and finite tests
    for name, val in [("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)]:
        if val < 0:
            raise ReplayValidationError(
                f"Negative value for {name} ({val}) in record for {symbol} at {ts}."
            )
        if not val.is_finite():
            raise ReplayValidationError(
                f"Non-finite value for {name} ({val}) in record for {symbol} at {ts}."
            )

    # OHLC structural invariants
    if h < o or h < c or h < l:
        raise ReplayValidationError(
            f"OHLC structure violation: High {h} must be greater than or equal to Open ({o}), Close ({c}), and Low ({l}) for {symbol} at {ts}."
        )
    if l > o or l > c:
        raise ReplayValidationError(
            f"OHLC structure violation: Low {l} must be less than or equal to Open ({o}) and Close ({c}) for {symbol} at {ts}."
        )
