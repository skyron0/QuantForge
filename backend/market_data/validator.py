from datetime import datetime
from decimal import Decimal
import math
from typing import Optional
from backend.market_data.exceptions import (
    MarketDataValidationError,
    StaleMarketDataError,
    FutureMarketDataError
)
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.models import (
    TradeTick, Candle, TickerSnapshot, OrderBookSnapshot
)


class MarketDataValidator:
    def __init__(self, policy: MarketDataPolicy) -> None:
        self.policy = policy

    def _validate_numeric(self, val: Decimal, field_name: str, allow_zero: bool = False) -> None:
        if val is None:
            raise MarketDataValidationError(f"Numeric validation failed for field '{field_name}': cannot be None")
        
        # Check float conversions for NaN/Inf safely
        try:
            val_float = float(val)
        except (ValueError, TypeError) as e:
            raise MarketDataValidationError(f"Numeric validation failed for field '{field_name}': invalid type {type(val)}") from e
        
        if math.isnan(val_float) or math.isinf(val_float):
            raise MarketDataValidationError(f"Numeric validation failed for field '{field_name}': cannot be NaN/Inf")
        
        if allow_zero:
            if val < Decimal("0"):
                raise MarketDataValidationError(f"Numeric validation failed for field '{field_name}': value {val} cannot be negative")
        else:
            if val <= Decimal("0"):
                raise MarketDataValidationError(f"Numeric validation failed for field '{field_name}': value {val} must be positive")

    def _validate_timestamp(self, ts_str: str, received_at_str: str) -> None:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            recv = datetime.fromisoformat(received_at_str.replace("Z", "+00:00"))
        except ValueError as e:
            raise MarketDataValidationError(f"Invalid timestamp format: {str(e)}") from e

        # Check stale limit based on max_market_data_age_seconds
        age = (recv - ts).total_seconds()
        if age > self.policy.max_market_data_age_seconds:
            raise StaleMarketDataError(
                f"Data is stale: timestamp {ts_str} is {age:.3f}s old, limit is {self.policy.max_market_data_age_seconds}s"
            )
        
        # Check future skew
        future_skew = (ts - recv).total_seconds()
        if future_skew > self.policy.max_future_skew_seconds:
            raise FutureMarketDataError(
                f"Data is in the future: timestamp {ts_str} is {future_skew:.3f}s in future, limit is {self.policy.max_future_skew_seconds}s"
            )

    def validate_trade(self, trade: TradeTick) -> None:
        if self.policy.allowed_symbols and trade.symbol not in self.policy.allowed_symbols:
            raise MarketDataValidationError(f"Symbol {trade.symbol} not in allowed symbols")
        self._validate_numeric(trade.price, "price")
        self._validate_numeric(trade.quantity, "quantity")
        if trade.side not in ("buy", "sell"):
            raise MarketDataValidationError(f"Side must be 'buy' or 'sell', got: {trade.side}")
        self._validate_timestamp(trade.timestamp, trade.received_at)

    def validate_candle(self, candle: Candle) -> None:
        if self.policy.allowed_symbols and candle.symbol not in self.policy.allowed_symbols:
            raise MarketDataValidationError(f"Symbol {candle.symbol} not in allowed symbols")
        self._validate_numeric(candle.open, "open")
        self._validate_numeric(candle.high, "high")
        self._validate_numeric(candle.low, "low")
        self._validate_numeric(candle.close, "close")
        self._validate_numeric(candle.volume, "volume", allow_zero=True)
        if candle.trade_count < 0:
            raise MarketDataValidationError("trade_count cannot be negative")

        # Open/High/Low/Close invariants
        if candle.low > candle.high:
            raise MarketDataValidationError(f"Candle low {candle.low} cannot exceed high {candle.high}")
        if candle.open > candle.high or candle.open < candle.low:
            raise MarketDataValidationError(f"Candle open {candle.open} must be between low {candle.low} and high {candle.high}")
        if candle.close > candle.high or candle.close < candle.low:
            raise MarketDataValidationError(f"Candle close {candle.close} must be between low {candle.low} and high {candle.high}")

        self._validate_timestamp(candle.open_time, candle.received_at)
        self._validate_timestamp(candle.close_time, candle.received_at)

    def validate_ticker(self, ticker: TickerSnapshot) -> None:
        if self.policy.allowed_symbols and ticker.symbol not in self.policy.allowed_symbols:
            raise MarketDataValidationError(f"Symbol {ticker.symbol} not in allowed symbols")
        self._validate_numeric(ticker.bid, "bid")
        self._validate_numeric(ticker.ask, "ask")
        self._validate_numeric(ticker.last, "last")
        self._validate_numeric(ticker.bid_quantity, "bid_quantity")
        self._validate_numeric(ticker.ask_quantity, "ask_quantity")
        self._validate_numeric(ticker.volume_24h, "volume_24h", allow_zero=True)
        
        # Check crossed book
        if not self.policy.allow_crossed_order_book:
            if ticker.bid >= ticker.ask:
                raise MarketDataValidationError(f"Crossed books not allowed: bid {ticker.bid} >= ask {ticker.ask}")
        self._validate_timestamp(ticker.timestamp, ticker.received_at)

    def validate_order_book(self, book: OrderBookSnapshot) -> None:
        if self.policy.allowed_symbols and book.symbol not in self.policy.allowed_symbols:
            raise MarketDataValidationError(f"Symbol {book.symbol} not in allowed symbols")
        
        prev_bid_price: Optional[Decimal] = None
        for i, level in enumerate(book.bids):
            self._validate_numeric(level.price, f"bids[{i}].price")
            self._validate_numeric(level.quantity, f"bids[{i}].quantity")
            if prev_bid_price is not None and level.price > prev_bid_price:
                raise MarketDataValidationError(f"Bids must be sorted descending: Level {i} price {level.price} > previous {prev_bid_price}")
            prev_bid_price = level.price

        prev_ask_price: Optional[Decimal] = None
        for i, level in enumerate(book.asks):
            self._validate_numeric(level.price, f"asks[{i}].price")
            self._validate_numeric(level.quantity, f"asks[{i}].quantity")
            if prev_ask_price is not None and level.price < prev_ask_price:
                raise MarketDataValidationError(f"Asks must be sorted ascending: Level {i} price {level.price} < previous {prev_ask_price}")
            prev_ask_price = level.price

        # Crossed check
        if book.bids and book.asks and not self.policy.allow_crossed_order_book:
            best_bid = book.bids[0].price
            best_ask = book.asks[0].price
            if best_bid >= best_ask:
                raise MarketDataValidationError(f"Crossed order book: best bid {best_bid} >= best ask {best_ask}")

        self._validate_timestamp(book.timestamp, book.received_at)
