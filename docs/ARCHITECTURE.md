\# QuantForge Architecture



\## Overview



QuantForge is a modular algorithmic trading platform.



Every module has a single responsibility.



Business logic must never be duplicated.



\---



\# High Level Pipeline



Market



↓



Collector



↓



Tick Queue



↓



Aggregator



↓



Consumer



↓



Indicator Engine



↓



Feature Engine



↓



Decision Engine



↓



Signal Validator



↓



Risk Manager



↓



Execution Engine



↓



Repositories



↓



Database



\---



\# Module Responsibilities



\## Collector



Receives real-time market data from exchanges.



Output:



Tick objects.



\---



\## Aggregator



Transforms ticks into OHLCV candles.



Output:



Completed Candle objects.



\---



\## Consumer



Owns the execution pipeline.



This is the central orchestration layer.



Every completed candle flows through this module.



No trading logic exists here.



Only orchestration.



\---



\## Indicator Engine



Calculates technical indicators.



Examples:



\- RSI

\- MACD

\- EMA

\- ATR

\- ADX

\- Bollinger Bands



Indicators are stateless.



\---



\## Feature Engine



Converts indicators into AI-ready features.



No trading decisions are made here.



\---



\## Decision Engine



Generates BUY / SELL / HOLD decisions.



Consumes feature vectors.



Does not execute trades.



\---



\## Signal Validator



Filters weak or invalid signals.



Responsible for execution quality.



\---



\## Risk Manager



Determines whether a position can be opened.



Responsible for:



\- Position sizing

\- Exposure

\- Risk limits



\---



\## Execution Engine



Responsible for order execution.



Current implementation:



Paper Trading



Future:



Live Exchange Execution



\---



\## Repositories



Persistence layer.



No business logic.



Only CRUD.



\---



\## Database



Stores:



\- ticks

\- candles

\- trades

\- snapshots



\---



\# Backtesting Philosophy



Backtesting must reuse the production pipeline.



Forbidden:



Separate trading logic.



Allowed:



Historical candles



↓



Consumer.process\_candle()



↓



Decision Engine



↓



Execution



\---



\# AI Philosophy



AI replaces only the Decision Engine.



Everything else remains identical.



Production



↓



Decision Engine



↓



Execution



AI



↓



AI Model



↓



Execution



Execution layer never changes.



\---



\# Design Principles



\- SOLID

\- DRY

\- Single Responsibility

\- Dependency Injection

\- Modular Design

\- Testability

\- Separation of Concerns



\---



\# Current Status



Architecture Stable



Major refactoring should be avoided unless necessary.

