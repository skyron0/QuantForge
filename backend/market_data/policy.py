from dataclasses import dataclass, field
from typing import Set
from backend.market_data.exceptions import InvalidMarketDataPolicyError


@dataclass(frozen=True)
class MarketDataPolicy:
    max_market_data_age_seconds: float = 5.0
    max_future_skew_seconds: float = 1.0
    max_sequence_gap: int = 0  # 0 indicates strict ordering (any gap raises sequence gap exception)
    reject_out_of_order_data: bool = True
    reject_duplicate_data: bool = True
    allow_sequence_reset: bool = False
    max_trade_buffer_size: int = 100
    max_candle_buffer_size: int = 200
    max_order_book_depth: int = 20
    max_symbols: int = 5
    allowed_symbols: Set[str] = field(default_factory=set)
    require_closed_candles_for_signals: bool = True
    allow_crossed_order_book: bool = False
    snapshot_ttl_seconds: float = 5.0
    provider_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_market_data_age_seconds <= 0:
            raise InvalidMarketDataPolicyError("max_market_data_age_seconds must be positive")
        if self.max_future_skew_seconds <= 0:
            raise InvalidMarketDataPolicyError("max_future_skew_seconds must be positive")
        if self.max_sequence_gap < 0:
            raise InvalidMarketDataPolicyError("max_sequence_gap cannot be negative")
        if self.max_trade_buffer_size <= 0:
            raise InvalidMarketDataPolicyError("max_trade_buffer_size must be positive")
        if self.max_candle_buffer_size <= 0:
            raise InvalidMarketDataPolicyError("max_candle_buffer_size must be positive")
        if self.max_order_book_depth <= 0:
            raise InvalidMarketDataPolicyError("max_order_book_depth must be positive")
        if self.max_symbols <= 0:
            raise InvalidMarketDataPolicyError("max_symbols must be positive")
        if self.snapshot_ttl_seconds <= 0:
            raise InvalidMarketDataPolicyError("snapshot_ttl_seconds must be positive")
        if self.provider_timeout_seconds <= 0:
            raise InvalidMarketDataPolicyError("provider_timeout_seconds must be positive")
