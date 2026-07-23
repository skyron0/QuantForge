import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from backend.runtime.event_bus import EventBus
from backend.runtime.events import TradingEvent
from backend.market_data.exceptions import MarketDataValidationError, StaleMarketDataError
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
from backend.market_data.bridge import (
    MarketDataBridge,
    MarketDataMessageReceived,
    MarketDataSnapshotUpdated,
)


def test_market_data_runtime_integration() -> None:
    """
    Verifies that raw market data is pushed through standard normalizer, validator,
    sequencing, and store layers, and finally gets published to EventBus asynchronously
    without loops.
    """
    # 1. Setup Event Bus
    bus = EventBus(max_queue_size=100)
    
    # 2. Setup Market Data components
    policy = MarketDataPolicy(
        allowed_symbols={"BTCUSDT", "ETHUSDT"},
        max_market_data_age_seconds=5.0,
        max_future_skew_seconds=2.0,
        max_sequence_gap=0,
    )
    store = MarketDataStore(policy)
    tracker = SequenceTracker(policy)
    normalizer = MarketDataNormalizer()
    validator = MarketDataValidator(policy)
    telemetry = MarketDataTelemetrySink()
    
    service = MarketDataService(
        normalizer=normalizer,
        validator=validator,
        sequence_tracker=tracker,
        store=store,
        telemetry=telemetry,
        policy=policy,
    )
    
    builder = MarketDataSnapshotBuilder(store, tracker, policy)
    bridge = MarketDataBridge(bus, service, builder)
    
    # 3. Register Event Bus Subscribers
    received_envelopes: List[TradingEvent] = []
    received_snapshots: List[TradingEvent] = []
    
    def on_envelope(event: TradingEvent) -> None:
        received_envelopes.append(event)
        
    def on_snapshot(event: TradingEvent) -> None:
        received_snapshots.append(event)
        
    bus.subscribe("MarketDataMessageReceived", on_envelope)
    bus.subscribe("MarketDataSnapshotUpdated", on_snapshot)
    
    # 4. Simulate raw provider event feed ingested and routed to EventBus
    now_str = datetime.now(timezone.utc).isoformat()
    raw_ticker = {
        "symbol": "BTC/USDT",
        "bid": 65000.0,
        "ask": 65010.0,
        "last": 65005.0,
        "bid_quantity": 0.5,
        "ask_quantity": 0.5,
        "timestamp": now_str,
        "sequence": 1,
    }
    
    raw_book = {
        "symbol": "BTC/USDT",
        "bids": [[65000.0, 1.0]],
        "asks": [[65010.0, 1.0]],
        "sequence": 1,
        "timestamp": now_str,
    }
    
    # Ingest through service
    envelope_ticker = service.ingest_raw_message("bybit", MarketDataType.TICKER, raw_ticker)
    envelope_book = service.ingest_raw_message("bybit", MarketDataType.ORDER_BOOK, raw_book)
    assert envelope_ticker.symbol == "BTCUSDT"
    assert envelope_book.symbol == "BTCUSDT"
    
    # Publish via bridge
    bridge.publish_envelope(envelope_ticker)
    bridge.publish_envelope(envelope_book)
    
    # Verify EventBus distribution
    assert len(received_envelopes) == 2
    
    # Check that store was updated
    stored_ticker = store.get_ticker("BTCUSDT")
    assert stored_ticker is not None
    assert stored_ticker.bid == Decimal("65000.0")
    
    # Build point-in-time snapshot and publish it
    snapshot = builder.build_snapshot("BTCUSDT", current_time_str=now_str)
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.ticker is not None
    assert snapshot.source_health == "CONNECTED"
    assert snapshot.metadata["is_valid_for_trading"] is True
    
    bridge.publish_snapshot(snapshot)
    assert len(received_snapshots) == 1
    received_snap_event = received_snapshots[0]
    assert isinstance(received_snap_event, MarketDataSnapshotUpdated)
    assert received_snap_event.snapshot is not None
    assert received_snap_event.snapshot.symbol == "BTCUSDT"


def test_stale_or_crossed_data_invalidates_snapshot_in_runtime() -> None:
    """
    Verifies that if stale ticker or crossed order book is ingested, the snapshot
    builder marks the point-in-time snapshot as STALE or DEGRADED, which makes it
    invalid for downstream trading authorization rules.
    """
    policy = MarketDataPolicy(
        allowed_symbols={"BTCUSDT"},
        max_market_data_age_seconds=5.0,
    )
    store = MarketDataStore(policy)
    tracker = SequenceTracker(policy)
    normalizer = MarketDataNormalizer()
    validator = MarketDataValidator(policy)
    telemetry = MarketDataTelemetrySink()
    
    service = MarketDataService(
        normalizer=normalizer,
        validator=validator,
        sequence_tracker=tracker,
        store=store,
        telemetry=telemetry,
        policy=policy,
    )
    builder = MarketDataSnapshotBuilder(store, tracker, policy)
    
    now = datetime.now(timezone.utc)
    
    # 1. Manually insert stale ticker directly to store to avoid ingestion time validation
    stale_time = now - timedelta(seconds=10)
    stale_ticker = TickerSnapshot(
        symbol="BTCUSDT",
        bid=Decimal("65000.0"),
        ask=Decimal("65010.0"),
        last=Decimal("65005.0"),
        bid_quantity=Decimal("0.5"),
        ask_quantity=Decimal("0.5"),
        volume_24h=Decimal("1000.0"),
        timestamp=stale_time.isoformat(),
        source="bybit",
        received_at=stale_time.isoformat(),
    )
    store.update_ticker(stale_ticker)
    
    # Also need an order book to not raise Missing critical order book SnapshotError
    fresh_book = OrderBookSnapshot(
        symbol="BTCUSDT",
        bids=[OrderBookLevel(price=Decimal("65000.0"), quantity=Decimal("1.0"))],
        asks=[OrderBookLevel(price=Decimal("65010.0"), quantity=Decimal("1.0"))],
        sequence=1,
        timestamp=now.isoformat(),
        source="bybit",
        received_at=now.isoformat(),
    )
    store.update_order_book(fresh_book)
    tracker.track("bybit", "BTCUSDT", MarketDataType.ORDER_BOOK, 1)
    
    # Now build snapshot relative to "now"
    snapshot = builder.build_snapshot("BTCUSDT", current_time_str=now.isoformat())
    assert snapshot.source_health == "STALE"
    assert snapshot.metadata["is_valid_for_trading"] is False

    # 2. Ingest order book gap -> marks order book sequence invalid -> DEGRADED snapshot
    # Reset store with fresh ticker
    raw_fresh_ticker = {
        "symbol": "BTCUSDT",
        "bid": 65000.0,
        "ask": 65010.0,
        "last": 65005.0,
        "bid_quantity": 0.5,
        "ask_quantity": 0.5,
        "timestamp": now.isoformat(),
        "sequence": 2,
    }
    service.ingest_raw_message("bybit", MarketDataType.TICKER, raw_fresh_ticker)
    
    # Ingest book with sequence 2 (no gap from 1)
    raw_book_1 = {
        "symbol": "BTCUSDT",
        "bids": [[65000.0, 1.0]],
        "asks": [[65010.0, 1.0]],
        "sequence": 2,
        "timestamp": now.isoformat(),
    }
    service.ingest_raw_message("bybit", MarketDataType.ORDER_BOOK, raw_book_1)
    
    # Gap jumps sequence to 4 (since gap limit is strict 0)
    raw_book_gap = {
        "symbol": "BTCUSDT",
        "bids": [[64998.0, 1.0]],
        "asks": [[65012.0, 1.0]],
        "sequence": 4,
        "timestamp": now.isoformat(),
    }

    # Ingestion will detect sequence gap and raise exception because policy max_sequence_gap=0
    # Let's catch it.
    with pytest.raises(Exception):
        service.ingest_raw_message("bybit", MarketDataType.ORDER_BOOK, raw_book_gap)
        
    # The active order book is marked invalid on gap check
    assert tracker.is_order_book_valid("bybit", "BTCUSDT") is False
    
    # Build snapshot -> should report DEGRADED health and not valid for trading
    snapshot_degraded = builder.build_snapshot("BTCUSDT", current_time_str=now.isoformat())
    assert snapshot_degraded.source_health == "DEGRADED"
    assert snapshot_degraded.metadata["is_valid_for_trading"] is False
