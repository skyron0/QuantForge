import threading
from typing import Any, Dict, Optional, Tuple
from backend.market_data.exceptions import (
    OutOfOrderMarketDataError,
    DuplicateMarketDataError,
    SequenceGapError
)
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.models import MarketDataType


class SequenceTracker:
    def __init__(self, policy: MarketDataPolicy) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        # Key: (provider, symbol, datatype) -> last_sequence_int
        self._sequences: Dict[Tuple[str, str, MarketDataType], int] = {}
        # Key: (provider, symbol) -> book_valid_bool
        self._order_book_validity: Dict[Tuple[str, str], bool] = {}

    def track(self, provider: str, symbol: str, data_type: MarketDataType, sequence: int) -> None:
        prov = provider.lower()
        sym = symbol.upper()
        key = (prov, sym, data_type)

        with self._lock:
            # First message received is always processed as base
            if key not in self._sequences:
                self._sequences[key] = sequence
                if data_type == MarketDataType.ORDER_BOOK:
                    self._order_book_validity[(prov, sym)] = True
                return

            last_seq = self._sequences[key]

            # Duplicate messages check
            if sequence == last_seq:
                if self.policy.reject_duplicate_data:
                    raise DuplicateMarketDataError(
                        f"Duplicate sequence message {sequence} detected for {prov}:{sym}:{data_type.value}"
                    )
                return

            # Out-of-order sequence check
            if sequence < last_seq:
                # Check for sequence reset (usually restarts at 0 or 1)
                if self.policy.allow_sequence_reset and sequence in (0, 1):
                    self._sequences[key] = sequence
                    if data_type == MarketDataType.ORDER_BOOK:
                        self._order_book_validity[(prov, sym)] = True
                    return

                if self.policy.reject_out_of_order_data:
                    raise OutOfOrderMarketDataError(
                        f"Out-of-order sequence {sequence} < last {last_seq} detected for {prov}:{sym}:{data_type.value}"
                    )
                return

            # Sequence gap check (greater than sequence jumps)
            gap = sequence - last_seq - 1
            if gap > 0:
                # Rejecting or invalidating order book
                if data_type == MarketDataType.ORDER_BOOK:
                    self._order_book_validity[(prov, sym)] = False

                if gap > self.policy.max_sequence_gap:
                    raise SequenceGapError(
                        f"Sequence gap detected: sequence jumped from {last_seq} to {sequence} (gap size = {gap}) for {prov}:{sym}:{data_type.value}"
                    )

            # Accept new sequence
            self._sequences[key] = sequence

    def invalidate_order_book(self, provider: str, symbol: str) -> None:
        with self._lock:
            self._order_book_validity[(provider.lower(), symbol.upper())] = False

    def is_order_book_valid(self, provider: str, symbol: str) -> bool:
        with self._lock:
            return self._order_book_validity.get((provider.lower(), symbol.upper()), True)

    def get_last_sequence(self, provider: str, symbol: str, data_type: MarketDataType) -> Optional[int]:
        with self._lock:
            return self._sequences.get((provider.lower(), symbol.upper(), data_type))

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                f"{k[0]}:{k[1]}:{k[2].value}": v
                for k, v in self._sequences.items()
            }
