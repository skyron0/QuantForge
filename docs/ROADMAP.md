# QuantForge Roadmap

Project Version: v0.3
Status: Active Development

---

# Vision

Build a professional AI-powered algorithmic trading platform with a reusable architecture, deterministic backtesting, and an AI decision engine trained on proprietary trading data.

---

# Development Strategy

Development follows milestone-based sprints.
Each sprint must end with:

- Working code
- Successful tests
- Updated documentation
- Git commit
- GitHub push
- No known regression

No sprint may leave the project in a broken state.

---

# Completed History

## Sprint 1: Infrastructure & Market Data ✅

- Docker, PostgreSQL, Redis Setup
- WebSocket Collector, Tick Queue, Candle Aggregator, Market Consumer
- Telemetry Logging & Basic Configurations

## Sprint 2: Core Trading Pipeline & Basic Engine ✅

- Indicator Engine & Decision Engine
- Signal Validator & Risk Manager
- Paper Executor & Portfolio Database Snapshots

## Sprint 3.0 - 3.10: Decoupled Runtime & Core Gates Foundational Milestone ✅

_Note: Individual sprint boundaries within the 3.0–3.10 range are not canonical and have been consolidated due to lack of historical repository partition evidence._

- Decoupled intelligence decision fusion with fast and slow signal paths
- Deterministic Risk Guard Engine enforcing drawdown/volatility gates
- Position Sizing Engine with risk-based sizing and unit normalization
- Execution Authorization Engine with idempotency and lineage safety
- Paper Execution Adapter simulating slippage, fees, and partial fills
- Portfolio Engine with Decimal-based accounting and position transitions
- Trailing Stop, Stop-Loss, and Take-Profit lifecycle states store
- PostgreSQL db migration adapters and persistence service bridge
- Bounded Market Data Store Coordinator

## Sprint 3.11: Feature Runtime & Online Inference Foundation ✅

- Causal feature extraction and historical feature buffers
- SHA-256 fingerprint feature schemas and validation bounds checks
- Model inference loader with strict compatibility gates

## Sprint 3.12: Paper Session CLI & Integration Tests ✅

- Dependency Injection Container (`QuantForgeContainer`) and health monitor
- Integrated paper session loops and developer CLI runner
- Continuous trading integrated mock tests

## Sprint 3.13: Historical Replay & Deterministic Simulation Runtime ✅

- Deterministic ReplayClock DI abstraction replacing wall-clock time
- Historical CSV candle loaders and ReplayScheduler
- Replay session results generation and telemetry integration

---

# Current Development

## Sprint 3.14: Advanced Performance Metrics & Analytics

**Status:** IN PROGRESS / FINALIZATION

### Current Verified State

- **Sharpe Ratio**: COMPLETE
- **Sortino Ratio**: COMPLETE
- **Equity Curve**: COMPLETE
- **Profit Factor**: PARTIAL — available in Backtest but missing/incomplete in Replay
- **Drawdown Analytics**: PARTIAL — base calculations exist; structured drawdown series persistence is missing
- **Cross-Runtime Sharpe/Sortino Semantics**: REQUIRES STANDARDIZATION — backtest currently calculates returns on trade-by-trade basis while replay uses period-over-period snapshot basis.

### Exit Criteria

- **Canonical Performance Measurement Policy**: Defines return sampling frequency (e.g. constant-interval period-over-period portfolio equity snapshots) for system metrics, separating closed-trade stats as a secondary metric category.
- **Consistent Time-Sampled Return Semantics**: Standardized calculations for Sharpe and Sortino ratios in both Backtest and Replay metrics runner.
- **Explicit Annualization Policy**: Correct $\sqrt{N}$ scaling logic for various sampling frequencies.
- **Profit Factor Availability**: Complete implementation in both backtest and replay contexts.
- **Drawdown Series Storage**: Structured drawdown analytics data saved for analytics database/dashboards.
- **Tests & Regression**: Deterministic unit tests for all edge cases (zero variance, zero trades, all wins/losses, NaN/Inf robustness) and full regression tests validation.

---

# Future Roadmap

## Sprint 3.15: Dataset Generation & Feature Schemas

**Status:** PLANNED

- Export features and labels pipelines from the backtest/replay engine.
- Feature/label set integrity verification with SHA-256 fingerprinting.
- Auto-validation of feature causality to prevent data leaks.

## Sprint 3.16: Model Training & Experiment Registry

**Status:** PLANNED

- Local Classical ML/Deep Learning training automation hooks.
- Model registry (file/database-backed catalog) with metadata tags.
- Simple experiment run performance tracker.

## Sprint 3.17: Simulation Realism & Risk Safeguards

**Status:** PLANNED

- Dynamic pricing models: spread, slippage, and execution latency simulation.
- Perpetual futures simulation: perpetual funding fees and leverage margin.
- Margin liquidation calculations in portfolio bookkeeping.

## Sprint 3.18: CCXT / Exchange Adapter Foundation

**Status:** PLANNED

- CCXT-adapter for Binance/Bybit spot and linear futures connectivity.
- Order state machine representing active/closed/cancelled orders on exchange.
- State reconciliation worker loop syncing local portfolio state and exchange wallet positions.

## Sprint 3.19: Operations Center, Alerts & Resilience

**Status:** PLANNED

- Telegram notification chatbot integration.
- Automated API WebSocket disconnection recovery loops.
- Failure injection testing & rate-limit safety guards.

## Sprint 3.20: Shadow Trading & Controlled Live Rollout

**Status:** PLANNED

- Shadow runtime execution mode (evaluating signals on live feeds without real capital orders).
- Strict guardian safety kill-switch limits.
- Gradual real capital deployment under manual safety gates.

## Later Sprints - Strategy Optimization & Advanced AI

**Status:** DEFERRED / RESEARCH

- Walk-forward parameter sweeps and genetic strategy optimization.
- Multi-agent hierarchical supervisor architecture.
- Real-time news/text sentiment pipeline extraction.

---

# Success Criteria

QuantForge is considered complete when:

- Live trading is stable and synced with exchange positions.
- Backtesting matches live execution closely across simulated slippage/latency thresholds.
- AI features are locally trainable and continuously validated for drift.
- Technical, quantitative, and operational safety rules are verified by an automated test suite.
