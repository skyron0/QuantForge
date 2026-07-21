import ast
import math
import threading
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pytest

from backend.execution_authorization.models import OrderDirection, OrderType, ExecutionEnvironment
from backend.execution_adapter.models import ExecutionResult, Fill, ExecutionStatus
from backend.portfolio.exceptions import (
    PortfolioValidationError,
    InvalidFillError,
    DuplicateFillError,
    PositionAccountingError,
    InsufficientPositionError,
    InvalidPositionTransitionError,
    PortfolioInvariantError,
    UnsupportedInstrumentError
)
from backend.portfolio.models import (
    PositionSide,
    Position,
    PortfolioState,
    PortfolioSnapshot
)
from backend.portfolio.policy import PortfolioPolicy
from backend.portfolio.idempotency import FillIdempotencyStore
from backend.portfolio.portfolio import PortfolioEngine
from backend.portfolio.bridge import PortfolioRiskContextBuilder
from backend.portfolio.telemetry import PortfolioTelemetrySink

# Mock Telemetry class
class MockPortfolioTelemetrySink(PortfolioTelemetrySink):
    def __init__(self):
        self.records = []
    
    def record_update(self, state, latency_ms, status, rejection_reason=""):
        self.records.append((state, latency_ms, status, rejection_reason))

# Test fixtures
@pytest.fixture
def policy():
    return PortfolioPolicy(
        policy_version="policy-v1",
        supported_instrument_types=["linear_perpetual", "spot"],
        allow_position_reversal=True,
        maximum_open_positions=5,
        maximum_symbol_positions=1,
        maximum_gross_exposure_fraction=Decimal("3.0"),
        maximum_net_exposure_fraction=Decimal("2.5"),
        maximum_leverage=Decimal("20.0"),
        market_price_max_age_seconds=60.0,
        maximum_future_clock_skew_seconds=10.0,
        accounting_tolerance=Decimal("0.001")
    )

@pytest.fixture
def idempotency_store():
    return FillIdempotencyStore(ttl_seconds=3600.0, max_capacity=20)

@pytest.fixture
def mock_telemetry():
    return MockPortfolioTelemetrySink()

@pytest.fixture
def engine(policy, idempotency_store, mock_telemetry):
    return PortfolioEngine(
        portfolio_id="port-123",
        initial_balance=Decimal("100000.0"),
        policy=policy,
        idempotency_store=idempotency_store,
        telemetry_sink=mock_telemetry
    )

def create_mock_result(fills, total_fees=0.0, leverage=1.0, completed_at=None):
    now_str = completed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return ExecutionResult(
        execution_id="exec-999",
        intent_id="intent-111",
        proposal_id="proposal-123",
        risk_authorization_id="auth-risk-456",
        sizing_id="sizing-789",
        symbol=fills[0].symbol if fills else "BTC/USDT",
        direction=OrderDirection.BUY if (fills and fills[0].direction == OrderDirection.BUY) else OrderDirection.SELL,
        requested_quantity=sum(f.quantity for f in fills) if fills else 0.0,
        filled_quantity=sum(f.quantity for f in fills) if fills else 0.0,
        average_fill_price=sum(f.price for f in fills)/len(fills) if fills else 0.0,
        total_notional=sum(f.quantity * f.price for f in fills) if fills else 0.0,
        total_fees=total_fees,
        total_slippage=0.0,
        status=ExecutionStatus.FILLED,
        fills=fills,
        rejection_reason="",
        adapter_name="paper",
        environment=ExecutionEnvironment.PAPER,
        started_at=now_str,
        completed_at=now_str,
        latency_ms=1.5,
        policy_version="exec-v1",
        metadata={"leverage": leverage}
    )

# 1. LONG Position Lifecycle Tests
def test_long_position_lifecycle(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # A. Open LONG
    fill1 = Fill(
        fill_id="f1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=2.0, price=50000.0, notional=100000.0, fee=10.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res1 = create_mock_result([fill1], total_fees=10.0, completed_at=completed_time)
    
    state = engine.apply_execution_result(res1)
    
    # Assert state after open LONG
    assert engine.get_position("BTC/USDT") is not None
    pos = engine.get_position("BTC/USDT")
    assert pos.side == PositionSide.LONG
    assert pos.quantity == Decimal("2.0")
    assert pos.average_entry_price == Decimal("50000.0")
    assert pos.accumulated_fees == Decimal("10.0")
    assert engine.get_state().cash_balance == Decimal("99990.0") # 100000 - 10.0 fee
    assert engine.get_state().total_fees == Decimal("10.0")

    # B. Add to LONG with weighted entry recalculation
    fill2 = Fill(
        fill_id="f2", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=53000.0, notional=53000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res2 = create_mock_result([fill2], total_fees=5.0, completed_at=completed_time)
    
    state = engine.apply_execution_result(res2)
    pos = engine.get_position("BTC/USDT")
    
    # Weighted avg entry = ((2 * 50000) + (1 * 53000)) / 3 = 153000 / 3 = 51000
    assert pos.quantity == Decimal("3.0")
    assert pos.average_entry_price == Decimal("51000.0")
    assert pos.accumulated_fees == Decimal("15.0")
    
    # C. Partial Close LONG
    fill3 = Fill(
        fill_id="f3", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.SELL,
        quantity=1.0, price=54000.0, notional=54000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res3 = create_mock_result([fill3], total_fees=5.0, completed_at=completed_time)
    
    state = engine.apply_execution_result(res3)
    pos = engine.get_position("BTC/USDT")
    
    # Realized PnL = (54000 - 51000) * 1.0 = +3000
    # Average entry preserved = 51000
    # Remaining quantity = 2.0
    assert pos.quantity == Decimal("2.0")
    assert pos.average_entry_price == Decimal("51000.0")
    assert engine.get_state().realized_pnl == Decimal("3000.0")
    
    # D. Full Close LONG
    fill4 = Fill(
        fill_id="f4", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.SELL,
        quantity=2.0, price=48000.0, notional=96000.0, fee=10.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res4 = create_mock_result([fill4], total_fees=10.0, completed_at=completed_time)
    
    state = engine.apply_execution_result(res4)
    # Realized PnL change = (48000 - 51000) * 2.0 = -6000
    # Total Realized PnL = 3000 - 6000 = -3000
    assert engine.get_position("BTC/USDT") is None
    assert engine.get_state().realized_pnl == Decimal("-3000.0")
    # Total fees = 10 + 5 + 5 + 10 = 30
    assert engine.get_state().total_fees == Decimal("30.0")
    # Cash balance = 100000 - 30 (fees) - 3000 (realized pnl) = 96970
    assert engine.get_state().cash_balance == Decimal("96970.0")
    assert engine.get_state().equity == Decimal("96970.0")

# 2. SHORT Position Lifecycle Tests
def test_short_position_lifecycle(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # Open SHORT
    fill1 = Fill(
        fill_id="sf1", intent_id="i1", symbol="ETH/USDT", direction=OrderDirection.SELL,
        quantity=10.0, price=3000.0, notional=30000.0, fee=3.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res1 = create_mock_result([fill1], total_fees=3.0, completed_at=completed_time)
    engine.apply_execution_result(res1)
    
    pos = engine.get_position("ETH/USDT")
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == Decimal("10.0")
    assert pos.average_entry_price == Decimal("3000.0")
    
    # Add to SHORT
    fill2 = Fill(
        fill_id="sf2", intent_id="i1", symbol="ETH/USDT", direction=OrderDirection.SELL,
        quantity=5.0, price=3300.0, notional=16500.0, fee=2.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res2 = create_mock_result([fill2], total_fees=2.0, completed_at=completed_time)
    engine.apply_execution_result(res2)
    
    pos = engine.get_position("ETH/USDT")
    # Avg Entry = ((10 * 3000) + (5 * 3300)) / 15 = 46500 / 15 = 3100.0
    assert pos.quantity == Decimal("15.0")
    assert pos.average_entry_price == Decimal("3100.0")
    
    # Partial Close SHORT (Profit)
    fill3 = Fill(
        fill_id="sf3", intent_id="i1", symbol="ETH/USDT", direction=OrderDirection.BUY,
        quantity=5.0, price=2800.0, notional=14000.0, fee=1.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    res3 = create_mock_result([fill3], total_fees=1.5, completed_at=completed_time)
    engine.apply_execution_result(res3)
    
    pos = engine.get_position("ETH/USDT")
    # PnL = (3100 - 2800) * 5.0 = +1500
    assert pos.quantity == Decimal("10.0")
    assert engine.get_state().realized_pnl == Decimal("1500.0")
    
    # Full Close SHORT (Loss)
    fill4 = Fill(
        fill_id="sf4", intent_id="i1", symbol="ETH/USDT", direction=OrderDirection.BUY,
        quantity=10.0, price=3400.0, notional=34000.0, fee=3.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    res4 = create_mock_result([fill4], total_fees=3.5, completed_at=completed_time)
    engine.apply_execution_result(res4)
    
    # PnL = (3100 - 3400) * 10 = -3000
    # Total Realized PnL = 1500 - 3000 = -1500
    assert engine.get_position("ETH/USDT") is None
    assert engine.get_state().realized_pnl == Decimal("-1500.0")
    assert engine.get_state().total_fees == Decimal("10.0")

# 3. Reversal tests
def test_position_reversal(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # Open LONG 1.0 unit
    fill1 = Fill(
        fill_id="rv1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill1], total_fees=5.0, completed_at=completed_time))
    
    # SELL 1.5 units (reversal to SHORT 0.5)
    fill2 = Fill(
        fill_id="rv2", intent_id="i2", symbol="BTC/USDT", direction=OrderDirection.SELL,
        quantity=1.5, price=52000.0, notional=78000.0, fee=7.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill2], total_fees=7.5, completed_at=completed_time))
    
    pos = engine.get_position("BTC/USDT")
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == Decimal("0.5")
    assert pos.average_entry_price == Decimal("52000.0")
    # PnL closed portion = (52000 - 50000) * 1.0 = +2000
    assert engine.get_state().realized_pnl == Decimal("2000.0")
    
    # Reversal to LONG 1.0 (requires BUY 1.5)
    fill3 = Fill(
        fill_id="rv3", intent_id="i3", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.5, price=51000.0, notional=76500.0, fee=7.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill3], total_fees=7.5, completed_at=completed_time))
    
    pos = engine.get_position("BTC/USDT")
    assert pos.side == PositionSide.LONG
    assert pos.quantity == Decimal("1.0")
    assert pos.average_entry_price == Decimal("51000.0")
    # PnL closed portion of SHORT = (52000 - 51000) * 0.5 = +500
    # Total PnL = 2000 + 500 = 2500
    assert engine.get_state().realized_pnl == Decimal("2500.0")

def test_position_reversal_disabled_rejection(policy, idempotency_store):
    strict_policy = PortfolioPolicy(
        policy_version="policy-v1",
        supported_instrument_types=["linear_perpetual"],
        allow_position_reversal=False, # disabled
        maximum_open_positions=5,
        maximum_symbol_positions=1,
        maximum_gross_exposure_fraction=Decimal("3.0"),
        maximum_net_exposure_fraction=Decimal("1.5"),
        maximum_leverage=Decimal("20.0"),
        market_price_max_age_seconds=60.0,
        maximum_future_clock_skew_seconds=10.0,
        accounting_tolerance=Decimal("0.001")
    )
    eng = PortfolioEngine("p-strict", Decimal("100000"), strict_policy, idempotency_store)
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    fill1 = Fill(
        fill_id="rvx1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    eng.apply_execution_result(create_mock_result([fill1], total_fees=5.0, completed_at=completed_time))
    
    fill2 = Fill(
        fill_id="rvx2", intent_id="i2", symbol="BTC/USDT", direction=OrderDirection.SELL,
        quantity=1.5, price=52000.0, notional=78000.0, fee=7.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    with pytest.raises(InvalidPositionTransitionError):
        eng.apply_execution_result(create_mock_result([fill2], total_fees=7.5, completed_at=completed_time))

# 4. Multi-Fills & Partial Fill Processing
def test_multiple_fills_atomic_application(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    fill1 = Fill(
        fill_id="mf1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=0.4, price=50000.0, notional=20000.0, fee=2.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    fill2 = Fill(
        fill_id="mf2", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=0.3, price=50010.0, notional=15003.0, fee=1.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    fill3 = Fill(
        fill_id="mf3", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=0.3, price=49990.0, notional=14997.0, fee=1.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    # Total sum fees = 5.0
    res = create_mock_result([fill1, fill2, fill3], total_fees=5.0, completed_at=completed_time)
    engine.apply_execution_result(res)
    
    pos = engine.get_position("BTC/USDT")
    assert pos.quantity == Decimal("1.0")
    # Avg Entry = ((0.4 * 50000) + (0.3 * 50010) + (0.3 * 49990)) / 1.0 = 50000.0
    assert pos.average_entry_price == Decimal("50000.0")
    assert engine.get_state().total_fees == Decimal("5.0")
    assert pos.source_fill_ids == ["mf1", "mf2", "mf3"]

# 5. Fee discrepancy and Rollback atomicity
def test_total_fee_mismatch_rejection(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="fmc1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    # total_fees in ExecutionResult is 10.0 but sum of fills is 5.0 (discrepancy > tolerance)
    res = create_mock_result([fill], total_fees=10.0, completed_at=completed_time)
    with pytest.raises(PortfolioInvariantError):
        engine.apply_execution_result(res)
    
    # Assert complete state rollback
    assert engine.get_position("BTC/USDT") is None
    assert engine.get_state().total_fees == Decimal("0")
    assert not engine.idempotency_store.is_processed("fmc1")

def test_atomic_application_partial_failure_rollback(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill1 = Fill(
        fill_id="at1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    fill2 = Fill(
        fill_id="at2", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    object.__setattr__(fill2, 'price', -10.0)
    res = create_mock_result([fill1, fill2], total_fees=10.0, completed_at=completed_time)
    
    with pytest.raises(InvalidFillError):
        engine.apply_execution_result(res)
        
    # Check that fill1 was not committed
    assert engine.get_position("BTC/USDT") is None
    assert engine.get_state().total_fees == Decimal("0")
    assert not engine.idempotency_store.is_processed("at1")

# 6. MTM / Realized vs Unrealized tests
def test_market_price_update_mtm(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="mtm1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=2.0, price=50000.0, notional=100000.0, fee=10.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill], total_fees=10.0, completed_at=completed_time))
    
    # Initial state
    assert engine.get_state().unrealized_pnl == Decimal("0")
    assert engine.get_state().equity == Decimal("99990.0")
    
    # Mark price updates to 51000
    update_time = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    engine.update_market_price("BTC/USDT", 51000.0, update_time)
    
    # MTM checks
    pos = engine.get_position("BTC/USDT")
    assert pos.current_price == Decimal("51000.0")
    assert pos.unrealized_pnl == Decimal("2000.0") # (51000 - 50000) * 2.0 = 2000
    assert engine.get_state().unrealized_pnl == Decimal("2000.0")
    assert engine.get_state().equity == Decimal("101990.0") # 99990 + 2000
    
    # Market update shouldn't mutate realized pnl
    assert engine.get_state().realized_pnl == Decimal("0")

# 7. Margin calculations and Leverage bounds
def test_margin_and_leverage(policy, idempotency_store):
    # Setup lev=10.0
    eng = PortfolioEngine("p-lev", Decimal("100000.0"), policy, idempotency_store)
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    fill = Fill(
        fill_id="lev1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=2.0, price=50000.0, notional=100000.0, fee=10.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    # Apply using leverage 10.0
    res = create_mock_result([fill], total_fees=10.0, leverage=10.0, completed_at=completed_time)
    eng.apply_execution_result(res)
    
    pos = eng.get_position("BTC/USDT")
    assert pos is not None
    assert pos.leverage == Decimal("10.0")
    # notional = 2.0 * 50000 = 100000
    # margin used = 100000 / 10 = 10000
    assert pos.margin_used == Decimal("10000.0")
    assert eng.get_state().used_margin == Decimal("10000.0")
    assert eng.get_state().available_balance == Decimal("89990.0") # 99990 - 10000

# 8. Exposure accounting (gross/net/symbol)
def test_exposures(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    fill1 = Fill(
        fill_id="exp1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    fill2 = Fill(
        fill_id="exp2", intent_id="i2", symbol="ETH/USDT", direction=OrderDirection.SELL,
        quantity=10.0, price=3000.0, notional=30000.0, fee=3.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill1], total_fees=5.0, completed_at=completed_time))
    engine.apply_execution_result(create_mock_result([fill2], total_fees=3.0, completed_at=completed_time))
    
    state = engine.get_state()
    # BTC LONG notional = 50000
    # ETH SHORT notional = 30000
    # Gross exposure = 50000 + 30000 = 80000
    # Net exposure = 50000 - 30000 = 20000
    assert state.gross_exposure == Decimal("80000.0")
    assert state.net_exposure == Decimal("20000.0")
    assert state.positions["BTC/USDT"].position_notional == Decimal("50000.0")

# 9. High-water mark & Drawdown
def test_high_water_mark_and_drawdown(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # 1. Start with initial equity = 100000. HWM = 100000
    assert engine.high_water_mark == Decimal("100000.0")
    assert engine.drawdown_fraction == Decimal("0")
    
    # 2. Add profitable trade
    fill = Fill(
        fill_id="hw1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=0.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill], total_fees=0.0, completed_at=completed_time))
    
    # Price updates to 60000. Equity = 110000. HWM should update to 110000
    update_time1 = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    engine.update_market_price("BTC/USDT", 60000.0, update_time1)
    
    assert engine.high_water_mark == Decimal("110000.0")
    assert engine.drawdown_fraction == Decimal("0")
    
    # 3. Price drops to 45000. Equity = 100000 - 5000 = 950000. HWM stays 110000
    update_time2 = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat().replace("+00:00", "Z")
    engine.update_market_price("BTC/USDT", 45000.0, update_time2)
    
    assert engine.high_water_mark == Decimal("110000.0")
    # Drawdown = (110000 - 95000) / 110000 = 15000 / 110000 = 0.136363
    assert math.isclose(float(engine.drawdown_fraction), 15000.0 / 110000.0)

# 10. Idempotency duplication and cleanup behavior
def test_duplicate_fill_ingestion_rejection(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="dup1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res = create_mock_result([fill], total_fees=5.0, completed_at=completed_time)
    
    # Initial
    engine.apply_execution_result(res)
    assert engine.get_position("BTC/USDT").quantity == Decimal("1.0")
    
    # Re-apply same fill. It will detect it as all duplicate and return state without duplicate mutation
    engine.apply_execution_result(res)
    assert engine.get_position("BTC/USDT").quantity == Decimal("1.0")

def test_fill_idempotency_store_eviction():
    store = FillIdempotencyStore(ttl_seconds=3600.0, max_capacity=3)
    
    store.record("f1")
    store.record("f2")
    store.record("f3")
    
    assert store.is_processed("f1")
    
    # record f4. Should evict f1
    store.record("f4")
    assert not store.is_processed("f1")
    assert store.is_processed("f4")

def test_fill_idempotency_concurrency():
    store = FillIdempotencyStore(ttl_seconds=3600.0, max_capacity=100)
    threads = []
    
    def worker(fill_id):
        store.record(fill_id)
        
    for i in range(50):
        t = threading.Thread(target=worker, args=(f"cf-{i}",))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    for i in range(50):
        assert store.is_processed(f"cf-{i}")

# 11. Bridge risk context building
def test_portfolio_risk_context_bridge(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="br1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=10.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    state = engine.apply_execution_result(create_mock_result([fill], total_fees=10.0, completed_at=completed_time))
    
    risk_ctx = PortfolioRiskContextBuilder.build_risk_context(
        state=state,
        symbol="BTC/USDT",
        volatility_state="NORMAL",
        market_liquidity_state="NORMAL",
        consecutive_losses=2,
        drawdown_fraction=engine.drawdown_fraction
    )
    
    # Assertions
    assert risk_ctx.symbol == "BTC/USDT"
    assert risk_ctx.equity == 99990.0
    assert risk_ctx.available_balance == 99990.0 - 50000.0 # equity - margin (leverage=1)
    assert risk_ctx.portfolio_exposure_pct == 50000.0 / 99990.0
    assert risk_ctx.symbol_exposure_pct == 50000.0 / 99990.0
    assert risk_ctx.volatility_state == "NORMAL"
    assert risk_ctx.consecutive_losses == 2

# 12. Snapshot & Lineage
def test_snapshots_lineage(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="snap1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill], total_fees=5.0, completed_at=completed_time))
    
    snap = engine.create_snapshot()
    assert isinstance(snap, PortfolioSnapshot)
    assert len(snap.open_positions) == 1
    assert snap.open_positions[0].source_fill_ids == ["snap1"]
    assert snap.open_positions[0].source_execution_ids == ["exec-999"]
    
    # Check immutability: positions dict in state shouldn't allow modification without raise
    state = engine.get_state()
    with pytest.raises(Exception):
        state.positions["ETH/USDT"] = None

# 13. Telemetry invocations
def test_telemetry_invocation(policy, idempotency_store, mock_telemetry):
    engine = PortfolioEngine(
        portfolio_id="telemetry-port",
        initial_balance=Decimal("10000.0"),
        policy=policy,
        idempotency_store=idempotency_store,
        telemetry_sink=mock_telemetry
    )
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill = Fill(
        fill_id="t1", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=0.1, price=50000.0, notional=5000.0, fee=2.5, slippage_amount=0.0,
        timestamp=completed_time
    )
    engine.apply_execution_result(create_mock_result([fill], total_fees=2.5, completed_at=completed_time))
    
    # Verify mock telemetry recorded the update
    assert len(mock_telemetry.records) == 1
    state, latency, status, reason = mock_telemetry.records[0]
    assert status == "SUCCESS"
    assert latency > 0.0

# 14. Invalid instrument, NaN, skew validations
def test_invalid_parameters_and_bounds(engine):
    completed_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    # Case A: Unsupported instrument type
    fill_bad_inst = Fill(
        fill_id="bad_inst", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time, metadata={"instrument_type": "option"}  # unsupported
    )
    res_bad = create_mock_result([fill_bad_inst], total_fees=5.0, completed_at=completed_time)
    with pytest.raises(UnsupportedInstrumentError):
        engine.apply_execution_result(res_bad)
        
    # Case B: NaN / Inf prices
    fill_nan = Fill(
        fill_id="nan_price", intent_id="i1", symbol="BTC/USDT", direction=OrderDirection.BUY,
        quantity=1.0, price=50000.0, notional=50000.0, fee=5.0, slippage_amount=0.0,
        timestamp=completed_time
    )
    res_nan = create_mock_result([fill_nan], total_fees=5.0, completed_at=completed_time)
    object.__setattr__(fill_nan, 'price', float("nan"))
    with pytest.raises(InvalidFillError):
        engine.apply_execution_result(res_nan)

# 15. AST dependency isolation validation
def test_portfolio_ast_dependencies():
    # Verify backend/portfolio module has ZERO imports of Ollama, reasoning engine, ccxt, or bourses
    import glob
    files = glob.glob("backend/portfolio/**/*.py", recursive=True)
    forbidden_terms = ["ollama", "OllamaProvider", "ReasoningEngine", "ccxt", "binance", "bybit", "requests", "httpx", "urllib"]
    
    for filename in files:
        with open(filename, "r", encoding="utf8") as f:
            code = f.read()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for term in forbidden_terms:
                            if term in alias.name.lower():
                                pytest.fail(f"Forbidden import '{alias.name}' detected in {filename}")
                elif isinstance(node, ast.ImportFrom):
                    for term in forbidden_terms:
                        if node.module and term in node.module.lower():
                            pytest.fail(f"Forbidden from-import module '{node.module}' detected in {filename}")
