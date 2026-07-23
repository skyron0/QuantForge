from datetime import datetime, timezone
from typing import Dict, List, Optional
from backend.market_data.exceptions import SnapshotError
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.models import MarketDataSnapshot
from backend.market_data.store import MarketDataStore
from backend.market_data.sequence import SequenceTracker


class MarketDataSnapshotBuilder:
    def __init__(
        self,
        store: MarketDataStore,
        sequence_tracker: SequenceTracker,
        policy: MarketDataPolicy
    ) -> None:
        self.store = store
        self.sequence_tracker = sequence_tracker
        self.policy = policy

    def build_snapshot(self, symbol: str, current_time_str: Optional[str] = None) -> MarketDataSnapshot:
        sym = symbol.upper()
        ticker = self.store.get_ticker(sym)
        order_book = self.store.get_order_book(sym)

        if not ticker:
            raise SnapshotError(f"Missing critical ticker snapshot for symbol: {sym}")
        if not order_book:
            raise SnapshotError(f"Missing critical order book snapshot for symbol: {sym}")

        # Gather candles maps safely
        candles_map = {}
        with self.store._lock:
            if sym in self.store._candles:
                for tf, dq in self.store._candles[sym].items():
                    candles_map[tf] = list(dq)

        # Get latest trade
        trades = self.store.get_trades(sym)
        latest_trade = trades[-1] if trades else None

        # Build sequence state dict
        seq_state = {}
        with self.sequence_tracker._lock:
            for key, val in self.sequence_tracker._sequences.items():
                if key[1] == sym:
                    seq_state[f"{key[0]}:{key[2].value}"] = val

        # Define time boundaries
        if current_time_str:
            try:
                eval_time = datetime.fromisoformat(current_time_str.replace("Z", "+00:00"))
            except ValueError as e:
                raise SnapshotError(f"Invalid current_time_str format: {str(e)}") from e
        else:
            eval_time = datetime.now(timezone.utc)

        try:
            ticker_ts = datetime.fromisoformat(ticker.timestamp.replace("Z", "+00:00"))
            book_ts = datetime.fromisoformat(order_book.timestamp.replace("Z", "+00:00"))
        except ValueError as e:
            raise SnapshotError(f"Failed to parse source timestamps: {str(e)}") from e

        ages = [
            (eval_time - ticker_ts).total_seconds(),
            (eval_time - book_ts).total_seconds()
        ]
        if latest_trade:
            try:
                trade_ts = datetime.fromisoformat(latest_trade.timestamp.replace("Z", "+00:00"))
                ages.append((eval_time - trade_ts).total_seconds())
            except ValueError as e:
                raise SnapshotError(f"Failed to parse trade timestamp: {str(e)}") from e

        max_age = max(ages)
        provider = ticker.source
        book_seq_valid = self.sequence_tracker.is_order_book_valid(provider, sym)

        is_stale = max_age > self.policy.max_market_data_age_seconds

        # Determine source_health and trading eligibility
        # If sequence is broken or data is stale, the snapshot is invalid for trading (fail closed).
        if not book_seq_valid or is_stale:
            # Marked as DEGRADED or STALE
            health = "DEGRADED" if not book_seq_valid else "STALE"
            is_valid_for_trading = False
        else:
            health = "CONNECTED"
            is_valid_for_trading = True

        timestamp_iso = eval_time.isoformat()

        return MarketDataSnapshot(
            symbol=sym,
            timestamp=timestamp_iso,
            ticker=ticker,
            latest_trade=latest_trade,
            candles=candles_map,
            order_book=order_book,
            source_health=health,
            data_age=max_age,
            sequence_state=seq_state,
            metadata={
                "is_valid_for_trading": is_valid_for_trading,
                "order_book_sequence_valid": book_seq_valid,
                "max_age_limit": self.policy.max_market_data_age_seconds
            }
        )
