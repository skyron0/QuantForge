from datetime import datetime, timezone
import uuid
from typing import Optional, Union, Dict, Any, List

from backend.market_data.exceptions import (
    NormalizationError,
    MarketDataValidationError,
    SequenceError,
    DuplicateMarketDataError,
    OutOfOrderMarketDataError,
    SequenceGapError
)
from backend.market_data.models import (
    MarketDataType,
    TradeTick,
    Candle,
    TickerSnapshot,
    OrderBookSnapshot,
    MarketDataEnvelope,
    MarketDataSnapshot
)
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.normalizer import MarketDataNormalizer
from backend.market_data.validator import MarketDataValidator
from backend.market_data.sequence import SequenceTracker
from backend.market_data.store import MarketDataStore
from backend.market_data.telemetry import MarketDataTelemetrySink


class MarketDataService:
    """
    Coordinator governing the ingestion boundary. Receives raw inputs, runs them
    through normalizer, validator, sequencer, and caches to memory store.
    """

    def __init__(
        self,
        normalizer: MarketDataNormalizer,
        validator: MarketDataValidator,
        sequence_tracker: SequenceTracker,
        store: MarketDataStore,
        telemetry: MarketDataTelemetrySink,
        policy: MarketDataPolicy
    ) -> None:
        self.normalizer = normalizer
        self.validator = validator
        self.sequence_tracker = sequence_tracker
        self.store = store
        self.telemetry = telemetry
        self.policy = policy

    def ingest_raw_message(
        self,
        provider: str,
        data_type: MarketDataType,
        raw_payload: Dict[str, Any]
    ) -> MarketDataEnvelope:
        """
        Main entry point for incoming streams. Normalizes -> Validates -> Checks Sequence -> Caches.
        Returns the canonical MarketDataEnvelope.
        """
        self.telemetry.record_received()
        start_time = datetime.now(timezone.utc)

        # 1. Normalization
        try:
            envelope = self._normalize_payload(provider, data_type, raw_payload)
        except NormalizationError as e:
            self.telemetry.record_validation_failure()
            raise e

        # 2. Validation
        try:
            self._validate_payload(envelope)
        except MarketDataValidationError as e:
            self.telemetry.record_validation_failure()
            raise e

        # 3. Sequencing
        try:
            self.sequence_tracker.track(
                provider=envelope.source,
                symbol=envelope.symbol,
                data_type=envelope.data_type,
                sequence=envelope.sequence
            )
        except DuplicateMarketDataError as e:
            self.telemetry.record_duplicate()
            raise e
        except OutOfOrderMarketDataError as e:
            self.telemetry.record_out_of_order()
            raise e
        except SequenceGapError as e:
            self.telemetry.record_sequence_gap()
            raise e

        # 4. Storage Caching
        self._store_payload(envelope)

        self.telemetry.record_accepted()
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.telemetry.record_latency(duration)

        return envelope

    def _normalize_payload(
        self,
        provider: str,
        data_type: MarketDataType,
        raw: Dict[str, Any]
    ) -> MarketDataEnvelope:
        raw_symbol = raw.get("symbol", "")
        symbol = self.normalizer.normalize_symbol(provider, raw_symbol)

        timestamp = raw.get("timestamp")
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        received_at = datetime.now(timezone.utc).isoformat()
        sequence = int(raw.get("sequence", 0))
        event_id = str(uuid.uuid4())

        payload: Union[TradeTick, TickerSnapshot, Candle, OrderBookSnapshot]

        if data_type == MarketDataType.TRADE:
            payload = TradeTick(
                symbol=symbol,
                price=self._to_decimal(raw["price"]),
                quantity=self._to_decimal(raw["quantity"]),
                side=self.normalizer.normalize_side(raw["side"]),
                trade_id=str(raw["trade_id"]),
                timestamp=timestamp,
                sequence=sequence,
                source=provider.lower(),
                received_at=received_at,
                metadata=raw.get("metadata", {})
            )
        elif data_type == MarketDataType.TICKER:
            payload = TickerSnapshot(
                symbol=symbol,
                bid=self._to_decimal(raw["bid"]),
                ask=self._to_decimal(raw["ask"]),
                last=self._to_decimal(raw["last"]),
                bid_quantity=self._to_decimal(raw["bid_quantity"]),
                ask_quantity=self._to_decimal(raw["ask_quantity"]),
                volume_24h=self._to_decimal(raw.get("volume_24h", 0)),
                timestamp=timestamp,
                source=provider.lower(),
                received_at=received_at,
                metadata=raw.get("metadata", {})
            )
        elif data_type == MarketDataType.CANDLE:
            payload = Candle(
                symbol=symbol,
                timeframe=self.normalizer.normalize_timeframe(raw["timeframe"]),
                open_time=raw["open_time"],
                close_time=raw["close_time"],
                open=self._to_decimal(raw["open"]),
                high=self._to_decimal(raw["high"]),
                low=self._to_decimal(raw["low"]),
                close=self._to_decimal(raw["close"]),
                volume=self._to_decimal(raw.get("volume", 0)),
                trade_count=int(raw.get("trade_count", 0)),
                closed=bool(raw.get("closed", True)),
                source=provider.lower(),
                sequence=sequence,
                received_at=received_at,
                metadata=raw.get("metadata", {})
            )
        elif data_type == MarketDataType.ORDER_BOOK:
            bids = [
                self.store.policy.max_order_book_depth if False else
                self._to_order_book_level(lvl) for lvl in raw.get("bids", [])
            ][:self.policy.max_order_book_depth]
            asks = [
                self._to_order_book_level(lvl) for lvl in raw.get("asks", [])
            ][:self.policy.max_order_book_depth]
            payload = OrderBookSnapshot(
                symbol=symbol,
                bids=bids,
                asks=asks,
                sequence=sequence,
                timestamp=timestamp,
                source=provider.lower(),
                received_at=received_at,
                metadata=raw.get("metadata", {})
            )
        else:
            raise NormalizationError(f"Unsupported market data type: {data_type}")

        return MarketDataEnvelope(
            event_id=event_id,
            data_type=data_type,
            symbol=symbol,
            source=provider.lower(),
            timestamp=timestamp,
            received_at=received_at,
            sequence=sequence,
            payload=payload
        )

    def _validate_payload(self, envelope: MarketDataEnvelope) -> None:
        p = envelope.payload
        if isinstance(p, TradeTick):
            self.validator.validate_trade(p)
        elif isinstance(p, TickerSnapshot):
            self.validator.validate_ticker(p)
        elif isinstance(p, Candle):
            self.validator.validate_candle(p)
        elif isinstance(p, OrderBookSnapshot):
            self.validator.validate_order_book(p)
        else:
            raise MarketDataValidationError("Unknown payload type")

    def _store_payload(self, envelope: MarketDataEnvelope) -> None:
        p = envelope.payload
        if isinstance(p, TradeTick):
            self.store.add_trade(p)
        elif isinstance(p, TickerSnapshot):
            self.store.update_ticker(p)
        elif isinstance(p, Candle):
            self.store.add_candle(p)
        elif isinstance(p, OrderBookSnapshot):
            self.store.update_order_book(p)

    def _to_decimal(self, val: Any) -> Any:
        from decimal import Decimal
        return Decimal(str(val))

    def _to_order_book_level(self, level: Union[Dict[str, Any], List[Any]]) -> Any:
        from backend.market_data.models import OrderBookLevel
        if isinstance(level, dict):
            return OrderBookLevel(
                price=self._to_decimal(level["price"]),
                quantity=self._to_decimal(level["quantity"])
            )
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            return OrderBookLevel(
                price=self._to_decimal(level[0]),
                quantity=self._to_decimal(level[1])
            )
        raise NormalizationError(f"Cannot parse order book level from: {level}")
