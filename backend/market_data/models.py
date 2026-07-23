from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MarketDataType(str, Enum):
    TRADE = "TRADE"
    TICKER = "TICKER"
    CANDLE = "CANDLE"
    ORDER_BOOK = "ORDER_BOOK"


@dataclass(frozen=True)
class TradeTick:
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str  # "buy" or "sell"
    trade_id: str
    timestamp: str  # ISO UTC
    sequence: int
    source: str
    received_at: str  # ISO UTC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candle:
    symbol: str
    timeframe: str  # Reuse existing conventions (e.g. "1m", "5m")
    open_time: str  # ISO UTC
    close_time: str  # ISO UTC
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trade_count: int
    closed: bool
    source: str
    sequence: int
    received_at: str  # ISO UTC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TickerSnapshot:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    bid_quantity: Decimal
    ask_quantity: Decimal
    volume_24h: Decimal
    timestamp: str  # ISO UTC
    source: str
    received_at: str  # ISO UTC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderBookLevel:
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence: int
    timestamp: str  # ISO UTC
    source: str
    received_at: str  # ISO UTC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataEnvelope:
    event_id: str
    data_type: MarketDataType
    symbol: str
    source: str
    timestamp: str  # ISO UTC
    received_at: str  # ISO UTC
    sequence: int
    payload: Union[TradeTick, TickerSnapshot, Candle, OrderBookSnapshot]


@dataclass(frozen=True)
class MarketDataSnapshot:
    symbol: str
    timestamp: str  # ISO UTC
    ticker: TickerSnapshot
    latest_trade: Optional[TradeTick]
    candles: Dict[str, List[Candle]]  # Key: timeframe (e.g. "1m") -> List of Candle entities
    order_book: OrderBookSnapshot
    source_health: str  # e.g., "CONNECTED", "DEGRADED", "DISCONNECTED"
    data_age: float  # seconds since oldest timestamp
    sequence_state: Dict[str, Any]  # dict containing sequence track info
    metadata: Dict[str, Any] = field(default_factory=dict)
