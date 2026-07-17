\# QuantForge Modules



This document describes the responsibility of every major module.



\---



\# backend/collector



Purpose



Receive raw market data.



Responsibilities



\- Connect to exchange

\- Receive ticks

\- Validate messages

\- Forward market data



Never



\- Calculate indicators

\- Execute trades



\---



\# backend/aggregator



Purpose



Convert ticks into OHLCV candles.



Input



Ticks



Output



Candles



Never



\- Make trading decisions



\---



\# backend/consumer



Purpose



Central orchestration layer.



Responsibilities



\- Receive completed candles

\- Execute trading pipeline

\- Coordinate modules



Never



\- Store business rules

\- Calculate indicators directly

\- Execute SQL



\---



\# backend/indicators



Purpose



Calculate technical indicators.



Examples



\- RSI

\- EMA

\- ATR

\- ADX

\- MACD

\- Bollinger Bands



Output



Indicator values



\---



\# backend/features



Purpose



Transform indicators into structured features.



Consumers



\- Rule-based Decision Engine

\- Future AI models



\---



\# backend/decision



Purpose



Generate BUY / SELL / HOLD signals.



Input



Feature vector



Output



Trading decision



Never



\- Execute orders



\---



\# backend/validation



Purpose



Validate trading signals.



Responsibilities



\- Confidence checks

\- Strategy filters

\- Duplicate prevention



\---



\# backend/risk



Purpose



Protect capital.



Responsibilities



\- Position sizing

\- Exposure limits

\- Risk validation



\---



\# backend/execution



Purpose



Execute approved orders.



Current



Paper trading



Future



Live exchange execution



\---



\# backend/repositories



Purpose



Database access layer.



Responsibilities



\- CRUD operations



Never



\- Trading logic

\- Risk logic

\- Strategy logic



\---



\# backend/models



Purpose



Database schema.



Contains



\- Tick

\- Candle

\- Trade

\- Portfolio

\- Snapshot



\---



\# dashboard



Purpose



Visualize system state.



Responsibilities



\- Trades

\- Portfolio

\- Statistics

\- Logs



Dashboard must never affect trading logic.



\---



\# Principle



Each module has one responsibility.



Communication between modules should happen through clearly defined interfaces.

