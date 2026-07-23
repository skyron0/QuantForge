import threading
from collections import deque
from typing import Dict, List, Optional
from backend.market_data.policy import MarketDataPolicy
from backend.market_data.models import (
    TradeTick, Candle, TickerSnapshot, OrderBookSnapshot
)


class MarketDataStore:
    def __init__(self, policy: MarketDataPolicy) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._tickers: Dict[str, TickerSnapshot] = {}
        self._order_books: Dict[str, OrderBookSnapshot] = {}
        self._trades: Dict[str, deque] = {}
        # Key: symbol -> {timeframe: deque of Candle}
        self._candles: Dict[str, Dict[str, deque]] = {}

    def update_ticker(self, ticker: TickerSnapshot) -> None:
        sym = ticker.symbol.upper()
        with self._lock:
            self._tickers[sym] = ticker

    def update_order_book(self, book: OrderBookSnapshot) -> None:
        sym = book.symbol.upper()
        with self._lock:
            self._order_books[sym] = book

    def add_trade(self, trade: TradeTick) -> None:
        sym = trade.symbol.upper()
        with self._lock:
            if sym not in self._trades:
                self._trades[sym] = deque(maxlen=self.policy.max_trade_buffer_size)
            self._trades[sym].append(trade)

    def add_candle(self, candle: Candle) -> None:
        sym = candle.symbol.upper()
        tf = candle.timeframe
        with self._lock:
            if sym not in self._candles:
                self._candles[sym] = {}
            if tf not in self._candles[sym]:
                self._candles[sym][tf] = deque(maxlen=self.policy.max_candle_buffer_size)
            self._candles[sym][tf].append(candle)

    def get_ticker(self, symbol: str) -> Optional[TickerSnapshot]:
        sym = symbol.upper()
        with self._lock:
            return self._tickers.get(sym)

    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        sym = symbol.upper()
        with self._lock:
            return self._order_books.get(sym)

    def get_trades(self, symbol: str) -> List[TradeTick]:
        sym = symbol.upper()
        with self._lock:
            if sym in self._trades:
                return list(self._trades[sym])
            return []

    def get_candles(self, symbol: str, timeframe: str) -> List[Candle]:
        sym = symbol.upper()
        with self._lock:
            if sym in self._candles and timeframe in self._candles[sym]:
                return list(self._candles[sym][timeframe])
            return []

    def clear(self, symbol: str) -> None:
        sym = symbol.upper()
        with self._lock:
            self._tickers.pop(sym, None)
            self._order_books.pop(sym, None)
            self._trades.pop(sym, None)
            self._candles.pop(sym, None)
