import ast
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest
from typing import Dict, Any

from backend.market_data.exceptions import (
    NormalizationError,
    MarketDataValidationError,
    DuplicateMarketDataError,
    OutOfOrderMarketDataError,
    SequenceGapError,
    StaleMarketDataError,
    FutureMarketDataError,
    SnapshotError,
)
from backend.market_data.models import (
    MarketDataType,
    TradeTick,
    Candle,
    TickerSnapshot,
    OrderBookLevel,
    OrderBookSnapshot,
)
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.normalizer import MarketDataNormalizer
from backend.market_data.validator import MarketDataValidator
from backend.market_data.sequence import SequenceTracker
from backend.market_data.store import MarketDataStore
from backend.market_data.snapshot import MarketDataSnapshotBuilder
from backend.market_data.telemetry import MarketDataTelemetrySink
from backend.market_data.service import MarketDataService
from backend.market_data.bridge import MarketDataBridge, MarketDataSnapshotUpdated
from backend.market_data.adapters.memory import MemoryMarketDataProvider
from backend.market_data.adapters.replay import ReplayMarketDataProvider
from backend.runtime.event_bus import BaseEventBus


class MockEventBus(BaseEventBus):
    def __init__(self) -> None:
        self.published = []

    def publish(self, event: Any) -> None:
        self.published.append(event)

    def subscribe(self, event_type: str, handler: Any) -> None:
        pass

    def unsubscribe(self, event_type: str, handler: Any) -> None:
        pass

    def clear(self) -> None:
        self.published.clear()


# ==========================================
# 1. Normalization & Code Design Constraints
# ==========================================

def test_centralized_normalizer() -> None:
    # Extensible registry
    norm = MarketDataNormalizer({
        "bybit": {"BTC/USDT": "BTCUSDT"}
    })
    
    assert norm.normalize_symbol("bybit", "BTC/USDT") == "BTCUSDT"
    # General cleanup uppercase replacements
    assert norm.normalize_symbol("other", "eth-usdt") == "ETHUSDT"
    assert norm.normalize_symbol("binance", "SOL_USDT") == "SOLUSDT"
    
    with pytest.raises(NormalizationError):
        norm.normalize_symbol("bybit", "")
        
    assert norm.normalize_side("buy") == "buy"
    assert norm.normalize_side("SHORT") == "sell"
    assert norm.normalize_timeframe("5min") == "5m"


# ==========================================
# 2. Strict Numeric & Structural Invariants
# ==========================================

def test_validator_numeric_bounds() -> None:
    policy = MarketDataPolicy(allowed_symbols={"BTCUSDT"})
    validator = MarketDataValidator(policy)

    # 1. NaN checks
    bad_tick = TickerSnapshot(
        symbol="BTCUSDT",
        bid=Decimal("NaN"),
        ask=Decimal("100"),
        last=Decimal("99"),
        bid_quantity=Decimal("1"),
        ask_quantity=Decimal("1"),
        volume_24h=Decimal("10"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="unit",
        received_at=datetime.now(timezone.utc).isoformat()
    )
    with pytest.raises(MarketDataValidationError):
        validator.validate_ticker(bad_tick)

    # 2. Inf checks
    bad_candle = Candle(
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=datetime.now(timezone.utc).isoformat(),
        close_time=datetime.now(timezone.utc).isoformat(),
        open=Decimal("100"),
        high=Decimal("Infinity"),
        low=Decimal("90"),
        close=Decimal("95"),
        volume=Decimal("10"),
        trade_count=10,
        closed=True,
        source="unit",
        sequence=1,
        received_at=datetime.now(timezone.utc).isoformat()
    )
    with pytest.raises(MarketDataValidationError):
        validator.validate_candle(bad_candle)

    # 3. Negative price checks
    bad_trade = TradeTick(
        symbol="BTCUSDT",
        price=Decimal("-100"),
        quantity=Decimal("1"),
        side="buy",
        trade_id="1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        sequence=1,
        source="unit",
        received_at=datetime.now(timezone.utc).isoformat()
    )
    with pytest.raises(MarketDataValidationError):
        validator.validate_trade(bad_trade)

    # 4. Crossed Order Book checks
    bad_book = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(Decimal("101"), Decimal("1"))],
        asks=[OrderBookLevel(Decimal("100"), Decimal("1"))], # Bid >= Ask
        sequence=1,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="unit",
        received_at=datetime.now(timezone.utc).isoformat()
    )
    with pytest.raises(MarketDataValidationError):
        validator.validate_order_book(bad_book)


# ==========================================
# 3. Sequence Tracking & Gap Detection
# ==========================================

def test_sequence_tracker_logic() -> None:
    policy = MarketDataPolicy(max_sequence_gap=0)  # strict ordering
    tracker = SequenceTracker(policy)

    # Base track
    tracker.track("unit", "BTCUSDT", MarketDataType.TRADE, 100)

    # Out of order
    with pytest.raises(OutOfOrderMarketDataError):
        tracker.track("unit", "BTCUSDT", MarketDataType.TRADE, 99)

    # Duplicate
    with pytest.raises(DuplicateMarketDataError):
        tracker.track("unit", "BTCUSDT", MarketDataType.TRADE, 100)

    # Gap
    with pytest.raises(SequenceGapError):
        tracker.track("unit", "BTCUSDT", MarketDataType.TRADE, 102) # Jumped 100 -> 102 (size=1)


def test_sequence_order_book_invalidation() -> None:
    policy = MarketDataPolicy(max_sequence_gap=5)  # gap up to 5 allowed, but book invalidates on any gap
    tracker = SequenceTracker(policy)

    tracker.track("unit", "BTCUSDT", MarketDataType.ORDER_BOOK, 10)
    assert tracker.is_order_book_valid("unit", "BTCUSDT") is True

    # Intentionally cause a gap to 13 (gap size = 2)
    tracker.track("unit", "BTCUSDT", MarketDataType.ORDER_BOOK, 13)
    # Order book validity must drop to False
    assert tracker.is_order_book_valid("unit", "BTCUSDT") is False


# ==========================================
# 4. Thread-Safe Store Capacity Boundaries
# ==========================================

def test_store_eviction_capacities() -> None:
    policy = MarketDataPolicy(
        max_trade_buffer_size=2,
        max_candle_buffer_size=3
    )
    store = MarketDataStore(policy)

    # Load trades
    t1 = TradeTick("BTC", Decimal("1"), Decimal("1"), "buy", "1", "ts", 1, "s", "recv")
    t2 = TradeTick("BTC", Decimal("2"), Decimal("1"), "buy", "2", "ts", 2, "s", "recv")
    t3 = TradeTick("BTC", Decimal("3"), Decimal("1"), "buy", "3", "ts", 3, "s", "recv")

    store.add_trade(t1)
    store.add_trade(t2)
    store.add_trade(t3)

    trades = store.get_trades("BTC")
    assert len(trades) == 2
    assert trades[0].trade_id == "2"
    assert trades[1].trade_id == "3"


# ==========================================
# 5. Snapshot Freshness & Stale/Degraded Flags
# ==========================================

def test_snapshot_freshness() -> None:
    policy = MarketDataPolicy(max_market_data_age_seconds=5.0)
    store = MarketDataStore(policy)
    tracker = SequenceTracker(policy)
    builder = MarketDataSnapshotBuilder(store, tracker, policy)

    now = datetime.now(timezone.utc)
    
    # Happy state
    ticker = TickerSnapshot(
        symbol="BTCUSDT", bid=Decimal("99"), ask=Decimal("100"), last=Decimal("99.5"),
        bid_quantity=Decimal("1"), ask_quantity=Decimal("1"), volume_24h=Decimal("10"),
        timestamp=now.isoformat(), source="unit", received_at=now.isoformat()
    )
    book = OrderBookSnapshot(
        symbol="BTCUSDT", bids=[OrderBookLevel(Decimal("99"), Decimal("10"))],
        asks=[OrderBookLevel(Decimal("100"), Decimal("10"))], sequence=10,
        timestamp=now.isoformat(), source="unit", received_at=now.isoformat()
    )
    
    store.update_ticker(ticker)
    store.update_order_book(book)
    tracker.track("unit", "BTCUSDT", MarketDataType.TICKER, 10)
    tracker.track("unit", "BTCUSDT", MarketDataType.ORDER_BOOK, 10)

    # Standard build
    snap = builder.build_snapshot("BTCUSDT", current_time_str=now.isoformat())
    assert snap.source_health == "CONNECTED"
    assert snap.metadata["is_valid_for_trading"] is True

    # 1. Stale ticker -> trading invalid, health STALE
    old_time = now - timedelta(seconds=10)
    stale_ticker = TickerSnapshot(
        symbol="BTCUSDT", bid=Decimal("99"), ask=Decimal("100"), last=Decimal("99.5"),
        bid_quantity=Decimal("1"), ask_quantity=Decimal("1"), volume_24h=Decimal("10"),
        timestamp=old_time.isoformat(), source="unit", received_at=now.isoformat()
    )
    store.update_ticker(stale_ticker)
    snap2 = builder.build_snapshot("BTCUSDT", current_time_str=now.isoformat())
    assert snap2.source_health == "STALE"
    assert snap2.metadata["is_valid_for_trading"] is False

    # Reset
    store.update_ticker(ticker)

    # 2. Sequence gap -> trading invalid, health DEGRADED
    tracker.invalidate_order_book("unit", "BTCUSDT")
    snap3 = builder.build_snapshot("BTCUSDT", current_time_str=now.isoformat())
    assert snap3.source_health == "DEGRADED"
    assert snap3.metadata["is_valid_for_trading"] is False


# ==========================================
# 6. Replay Provider Determinism
# ==========================================

def test_replay_provider_stepping() -> None:
    policy = MarketDataPolicy(max_market_data_age_seconds=100.0)
    store = MarketDataStore(policy)
    tracker = SequenceTracker(policy)
    norm = MarketDataNormalizer()
    validator = MarketDataValidator(policy)
    telemetry = MarketDataTelemetrySink()
    service = MarketDataService(norm, validator, tracker, store, telemetry, policy)

    replay = ReplayMarketDataProvider(service)
    replay.subscribe("BTCUSDT")
    replay.start()

    # Preload manual events
    now_str = datetime.now(timezone.utc).isoformat()
    events_mock = [
        {
            "data_type": "TICKER",
            "symbol": "BTCUSDT",
            "payload": {
                "symbol": "BTCUSDT", "bid": 98.0, "ask": 99.0, "last": 98.5,
                "bid_quantity": 1, "ask_quantity": 1, "volume_24h": 100,
                "sequence": 1, "timestamp": now_str
            }
        },
        {
            "data_type": "ORDER_BOOK",
            "symbol": "BTCUSDT",
            "payload": {
                "symbol": "BTCUSDT",
                "bids": [[98.0, 10]], "asks": [[99.0, 10]],
                "sequence": 1, "timestamp": now_str
            }
        }
    ]

    replay.load_data(events_mock)
    assert replay.has_next() is True

    env1 = replay.step()
    assert env1 is not None
    assert env1.data_type == MarketDataType.TICKER
    ticker_val = store.get_ticker("BTCUSDT")
    assert ticker_val is not None
    assert ticker_val.bid == Decimal("98.0")

    env2 = replay.step()
    assert env2 is not None
    assert env2.data_type == MarketDataType.ORDER_BOOK
    book_val = store.get_order_book("BTCUSDT")
    assert book_val is not None
    assert book_val.bids[0].price == Decimal("98.0")

    assert replay.step() is None


# ==========================================
# 7. EventBus & Bridge Integration
# ==========================================

def test_bridge_publishes_to_event_bus() -> None:
    bus = MockEventBus()
    policy = MarketDataPolicy()
    store = MarketDataStore(policy)
    tracker = SequenceTracker(policy)
    norm = MarketDataNormalizer()
    validator = MarketDataValidator(policy)
    telemetry = MarketDataTelemetrySink()
    service = MarketDataService(norm, validator, tracker, store, telemetry, policy)
    builder = MarketDataSnapshotBuilder(store, tracker, policy)

    bridge = MarketDataBridge(bus, service, builder)

    # Ingest ticker env
    now_str = datetime.now(timezone.utc).isoformat()
    raw = {
        "symbol": "BTCUSDT", "bid": 99, "ask": 100, "last": 99.5,
        "bid_quantity": 1, "ask_quantity": 1, "sequence": 10, "timestamp": now_str
    }
    env = service.ingest_raw_message("unit", MarketDataType.TICKER, raw)

    bridge.publish_envelope(env)
    assert len(bus.published) == 1
    assert bus.published[0].envelope.symbol == "BTCUSDT"


# ==========================================
# 8. Dependency Isolation Safeguard (AST check)
# ==========================================

def test_market_data_dependency_isolation() -> None:
    """Verifies that backend/market_data files contain zero references to SQLAlchemy or Ollama."""
    import os
    package_dir = os.path.dirname(os.path.dirname(__file__)) # points to backend
    market_data_dir = os.path.join(package_dir, "market_data")

    forbidden_terms = ["sqlalchemy", "ollama", "risk_guard", "PositionSizingEngine"]

    for root, _, files in os.walk(market_data_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    parsed = ast.parse(content, filename=path)
                    for node in ast.walk(parsed):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                for term in forbidden_terms:
                                    if term.lower() in alias.name.lower():
                                        pytest.fail(f"Forbidden import '{alias.name}' containing term '{term}' in {path}")
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                for term in forbidden_terms:
                                    if term.lower() in node.module.lower():
                                        pytest.fail(f"Forbidden from-import module '{node.module}' containing term '{term}' in {path}")
