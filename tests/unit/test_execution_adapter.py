import os
import ast
import math
import uuid
import pytest
import threading
import time
import dataclasses
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from backend.execution_authorization.models import OrderIntent, ExecutionEnvironment, OrderDirection, OrderType
from backend.execution_adapter.exceptions import (
    ExecutionAdapterError,
    ExecutionAdapterValidationError,
    UnsupportedExecutionEnvironmentError,
    UnsupportedOrderTypeError,
    StaleExecutionContextError,
    InsufficientLiquidityError,
    DuplicateExecutionError,
    InvalidMarketStateError,
    ExecutionSimulationError
)
from backend.execution_adapter.models import (
    ExecutionStatus,
    Fill,
    ExecutionResult,
    PaperExecutionContext
)
from backend.execution_adapter.policy import PaperExecutionPolicy
from backend.execution_adapter.idempotency import ExecutionIdempotencyStore
from backend.execution_adapter.paper import PaperExecutionAdapter
from backend.execution_adapter.telemetry import ExecutionTelemetrySink

# 1. Test Fixtures
@pytest.fixture
def base_policy():
    return PaperExecutionPolicy(
        policy_version="exec-policy-v1",
        maximum_market_data_age_seconds=10.0,
        maximum_future_clock_skew_seconds=2.0,
        fee_rate=0.001,       # 0.1%
        slippage_rate=0.0005,  # 0.05%
        allow_partial_fills=True,
        minimum_fill_quantity=0.0001,
        reject_if_insufficient_liquidity=False,
        intent_max_age_seconds=60.0,
        execution_result_ttl_seconds=3600.0,
    )

@pytest.fixture
def base_idempotency_store():
    return ExecutionIdempotencyStore(ttl_seconds=60.0)

class MockTelemetrySink(ExecutionTelemetrySink):
    def __init__(self):
        self.records = []

    def record(self, result: ExecutionResult, latency_ms: float):
        self.records.append((result, latency_ms))

@pytest.fixture
def mock_telemetry():
    return MockTelemetrySink()

@pytest.fixture
def adapter(base_policy, base_idempotency_store, mock_telemetry):
    return PaperExecutionAdapter(
        policy=base_policy,
        idempotency_store=base_idempotency_store,
        telemetry_sink=mock_telemetry
    )

@pytest.fixture
def base_intent():
    now_dt = datetime.now(timezone.utc)
    # Valid OrderIntent
    return OrderIntent(
        intent_id="intent-111",
        idempotency_key="idempotency-xxx",
        proposal_id="proposal-123",
        risk_authorization_id="auth-risk-456",
        sizing_id="sizing-789",
        symbol="BTC/USDT",
        direction=OrderDirection.BUY,
        quantity=0.1,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_loss=49000.0,
        take_profit=52000.0,
        environment=ExecutionEnvironment.PAPER,
        source_model_version="ml-v1",
        fusion_policy_version="fusion-v1",
        risk_policy_version="risk-v1",
        position_sizing_policy_version="sizing-v1",
        execution_policy_version="auth-policy-v1",
        reasoning_request_id="req-999",
        created_at=(now_dt - timedelta(seconds=10)).isoformat().replace("+00:00", "Z"),
        expires_at=(now_dt + timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
    )

@pytest.fixture
def base_context():
    now_dt = datetime.now(timezone.utc)
    # Valid PaperExecutionContext
    return PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=100.0,
        timestamp=now_dt.isoformat().replace("+00:00", "Z")
    )

# 2. Market Execution Tests
def test_valid_buy_market_execution(adapter, base_intent, base_context, mock_telemetry):
    # BUY MARKET uses ask_price (50010.0)
    # Execution price = ask_price * (1 + 0.0005) = 50010.0 * 1.0005 = 50035.005
    # Quantity = 0.1
    # Notional = 0.1 * 50035.005 = 5003.5005
    # Slippage = 0.1 * 50010.0 * 0.0005 = 2.5005
    # Fee = 5003.5005 * 0.001 = 5.0035005
    res = adapter.execute(base_intent, base_context)
    
    assert res.status == ExecutionStatus.FILLED
    assert res.filled_quantity == 0.1
    assert math.isclose(res.average_fill_price, 50035.005)
    assert math.isclose(res.total_notional, 5003.5005)
    assert math.isclose(res.total_slippage, 2.5005)
    assert math.isclose(res.total_fees, 5.0035005)
    assert len(res.fills) == 1
    assert len(mock_telemetry.records) == 1

def test_valid_sell_market_execution(adapter, base_intent, base_context):
    sell_intent = dataclasses.replace(base_intent, direction=OrderDirection.SELL)
    # SELL MARKET uses bid_price (49990.0)
    # Price = 49990.0 * (1 - 0.0005) = 49965.005
    # Quantity = 0.1
    res = adapter.execute(sell_intent, base_context)
    
    assert res.status == ExecutionStatus.FILLED
    assert res.filled_quantity == 0.1
    assert math.isclose(res.average_fill_price, 49965.005)
    assert math.isclose(res.total_slippage, 2.4995)

# 3. Limit Execution Tests
def test_valid_buy_limit_fill(adapter, base_intent, base_context):
    limit_intent = dataclasses.replace(
        base_intent,
        order_type=OrderType.LIMIT,
        limit_price=50050.0  # Ask is 50010.0 <= limit price (met)
    )
    res = adapter.execute(limit_intent, base_context)
    assert res.status == ExecutionStatus.FILLED
    assert res.average_fill_price == 50050.0
    assert res.total_slippage == 0.0  # limit orders have 0 slippage

def test_valid_sell_limit_fill(adapter, base_intent, base_context):
    limit_intent = dataclasses.replace(
        base_intent,
        direction=OrderDirection.SELL,
        order_type=OrderType.LIMIT,
        limit_price=49950.0  # Bid is 49990.0 >= limit price (met)
    )
    res = adapter.execute(limit_intent, base_context)
    assert res.status == ExecutionStatus.FILLED
    assert res.average_fill_price == 49950.0

def test_unfilled_buy_limit_behavior(adapter, base_intent, base_context):
    limit_intent = dataclasses.replace(
        base_intent,
        order_type=OrderType.LIMIT,
        limit_price=49900.0  # Ask is 50010.0 > 49900.0 (condition not met)
    )
    res = adapter.execute(limit_intent, base_context)
    assert res.status == ExecutionStatus.ACCEPTED
    assert res.filled_quantity == 0.0
    assert "Limit price condition not met" in res.rejection_reason

def test_unfilled_sell_limit_behavior(adapter, base_intent, base_context):
    limit_intent = dataclasses.replace(
        base_intent,
        direction=OrderDirection.SELL,
        order_type=OrderType.LIMIT,
        limit_price=50100.0  # Bid is 49990.0 < 50100.0 (condition not met)
    )
    res = adapter.execute(limit_intent, base_context)
    assert res.status == ExecutionStatus.ACCEPTED
    assert res.filled_quantity == 0.0

# 4. Partial Fills and Liquidity constraints
def test_partial_fill(adapter, base_intent):
    # Context liquidity is 0.05, but intent wants 0.1
    small_liq_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=0.05,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    res = adapter.execute(base_intent, small_liq_context)
    assert res.status == ExecutionStatus.PARTIALLY_FILLED
    assert res.filled_quantity == 0.05

def test_insufficient_liquidity_rejection(adapter, base_intent):
    # Policy has reject_if_insufficient_liquidity = True
    strict_policy = PaperExecutionPolicy(
        policy_version="exec-policy-strict",
        reject_if_insufficient_liquidity=True
    )
    strict_adapter = PaperExecutionAdapter(policy=strict_policy, idempotency_store=ExecutionIdempotencyStore())
    
    small_liq_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=0.05,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    res = strict_adapter.execute(base_intent, small_liq_context)
    assert res.status == ExecutionStatus.REJECTED
    assert res.filled_quantity == 0.0
    assert "Insufficient available market liquidity" in res.rejection_reason

def test_partial_fills_disabled(adapter, base_intent):
    # Policy has allow_partial_fills = False
    no_part_policy = PaperExecutionPolicy(
        policy_version="exec-policy-no-part",
        allow_partial_fills=False
    )
    no_part_adapter = PaperExecutionAdapter(policy=no_part_policy, idempotency_store=ExecutionIdempotencyStore())
    
    small_liq_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=0.05,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    res = no_part_adapter.execute(base_intent, small_liq_context)
    assert res.status == ExecutionStatus.REJECTED
    assert res.filled_quantity == 0.0

def test_filled_quantity_never_exceeds_requested_quantity(adapter, base_intent, base_context):
    # Available liquidity is 100.0, but intent requested only 0.1
    res = adapter.execute(base_intent, base_context)
    assert res.filled_quantity <= base_intent.quantity

def test_filled_quantity_never_exceeds_available_liquidity(adapter, base_intent):
    small_liq_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=0.05,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    res = adapter.execute(base_intent, small_liq_context)
    assert res.filled_quantity <= small_liq_context.available_liquidity

# 5. Safety validation tests
def test_stale_market_context_rejection(adapter, base_intent):
    # clock skew in the past: context is offset by 20 seconds, exceeding maximum age of 10s
    stale_dt = datetime.now(timezone.utc) - timedelta(seconds=20)
    stale_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=100.0,
        timestamp=stale_dt.isoformat().replace("+00:00", "Z")
    )
    with pytest.raises(StaleExecutionContextError):
        adapter.execute(base_intent, stale_context)

def test_future_timestamp_rejection(adapter, base_intent):
    # clock skew in the future: context timestamp is +10 seconds in the future
    future_dt = datetime.now(timezone.utc) + timedelta(seconds=10)
    future_context = PaperExecutionContext(
        current_market_price=50000.0,
        bid_price=49990.0,
        ask_price=50010.0,
        available_liquidity=100.0,
        timestamp=future_dt.isoformat().replace("+00:00", "Z")
    )
    with pytest.raises(StaleExecutionContextError):
        adapter.execute(base_intent, future_context)

def test_expired_order_intent_rejection(adapter, base_intent, base_context):
    # Intent expires before the context timestamp
    expired_dt = datetime.now(timezone.utc) - timedelta(seconds=20)
    expired_intent = dataclasses.replace(
        base_intent, expires_at=expired_dt.isoformat().replace("+00:00", "Z")
    )
    res = adapter.execute(expired_intent, base_context)
    assert res.status == ExecutionStatus.REJECTED
    assert "expired" in res.rejection_reason

def test_live_intent_rejection(adapter, base_intent, base_context):
    live_intent = dataclasses.replace(base_intent, environment=ExecutionEnvironment.LIVE)
    with pytest.raises(UnsupportedExecutionEnvironmentError):
        adapter.execute(live_intent, base_context)

def test_shadow_intent_rejection(adapter, base_intent, base_context):
    shadow_intent = dataclasses.replace(base_intent, environment=ExecutionEnvironment.SHADOW)
    with pytest.raises(UnsupportedExecutionEnvironmentError):
        adapter.execute(shadow_intent, base_context)

# 6. Invalid Parameter Tests
def test_invalid_nan_inf_values(base_intent):
    with pytest.raises(ValueError):
        PaperExecutionContext(
            current_market_price=float('nan'),
            bid_price=49990.0,
            ask_price=50010.0,
            available_liquidity=100.0,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    with pytest.raises(ValueError):
        PaperExecutionContext(
            current_market_price=50000.0,
            bid_price=49990.0,
            ask_price=float('inf'),
            available_liquidity=100.0,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

def test_invalid_bid_ask_state():
    with pytest.raises(ValueError):
        PaperExecutionContext(
            current_market_price=50000.0,
            bid_price=50010.0,  # bid > ask (invalid)
            ask_price=49990.0,
            available_liquidity=100.0,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

# 7. Lineage and Policy preservation
def test_lineage_preservation(adapter, base_intent, base_context):
    res = adapter.execute(base_intent, base_context)
    assert res.intent_id == base_intent.intent_id
    assert res.proposal_id == base_intent.proposal_id
    assert res.risk_authorization_id == base_intent.risk_authorization_id
    assert res.sizing_id == base_intent.sizing_id
    assert res.policy_version == adapter.policy.policy_version

# 8. Duplicate / Idempotency tests
def test_duplicate_execution_rejection(adapter, base_intent, base_context):
    res1 = adapter.execute(base_intent, base_context)
    assert res1.status == ExecutionStatus.FILLED
    
    # Try again with same intent
    res2 = adapter.execute(base_intent, base_context)
    assert res2.execution_id == res1.execution_id
    assert res2.status == ExecutionStatus.FILLED

def test_idempotency_rollback_on_failure(adapter, base_intent, base_context):
    # Send live intent which will fail validation
    live_intent = dataclasses.replace(base_intent, environment=ExecutionEnvironment.LIVE)
    with pytest.raises(UnsupportedExecutionEnvironmentError):
        adapter.execute(live_intent, base_context)
    
    # Claim should be released. Verify we can claim it now for a PAPER intent with same ID
    paper_intent = dataclasses.replace(base_intent, intent_id=live_intent.intent_id)
    res = adapter.execute(paper_intent, base_context)
    assert res.status == ExecutionStatus.FILLED

def test_concurrent_duplicate_execution_safety(adapter, base_intent, base_context):
    threads = []
    results = []
    errors = []

    def run_worker():
        try:
            res = adapter.execute(base_intent, base_context)
            results.append(res)
        except DuplicateExecutionError as e:
            errors.append(e)
        except Exception as e:
            pass

    for _ in range(10):
        t = threading.Thread(target=run_worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Out of 10 threads, only 1 should be authorized first, and the rest either
    # get the cached return result (ExecutionResult) or a DuplicateExecutionError.
    # But because they use the same idempotency store, the ones that execute
    # concurrently will raise DuplicateExecutionError if the query is in progress,
    # or return the identical result if we check while completed.
    # The key point: only one execution should succeed (create the actual execution).
    # Since they return the cached identical object or fail, let's verify execution_ids
    exec_ids = {r.execution_id for r in results}
    assert len(exec_ids) == 1

def test_bounded_idempotency_memory_behavior(base_intent, base_context):
    # Set capacity to 3
    store = ExecutionIdempotencyStore(ttl_seconds=3600.0, max_capacity=3)
    policy = PaperExecutionPolicy(policy_version="exec-v1")
    exec_adapter = PaperExecutionAdapter(policy=policy, idempotency_store=store)

    intent1 = dataclasses.replace(base_intent, intent_id="intent-1")
    intent2 = dataclasses.replace(base_intent, intent_id="intent-2")
    intent3 = dataclasses.replace(base_intent, intent_id="intent-3")
    intent4 = dataclasses.replace(base_intent, intent_id="intent-4")

    # Fill 3
    exec_adapter.execute(intent1, base_context)
    exec_adapter.execute(intent2, base_context)
    exec_adapter.execute(intent3, base_context)

    assert store.get_result("intent-1") is not None

    # Adding 4th should kick out intent-1 (as capacity=3 is exceeded)
    exec_adapter.execute(intent4, base_context)
    
    assert store.get_result("intent-1") is None
    assert store.get_result("intent-4") is not None

# 9. Architectural Isolation Test
def test_architecture_isolation_ast_tests():
    """
    Scans the backend/execution_adapter package files to guarantee zero dependencies
    on: CCXT, exchange SDKs, Ollama, OpenAI, or LLMs.
    """
    forbidden_imports = {
        "ccxt", "binance", "bybit", "ollama", "openai", "anthropic", "langchain",
        "requests", "urllib", "websocket"
    }

    dir_to_scan = os.path.join("backend", "execution_adapter")
    assert os.path.exists(dir_to_scan), "execution_adapter package directory does not exist"

    for root, _, files in os.walk(dir_to_scan):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            root_module = alias.name.split(".")[0]
                            assert root_module not in forbidden_imports, (
                                f"Forbidden import '{alias.name}' detected in {path}"
                            )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            root_module = node.module.split(".")[0]
                            assert root_module not in forbidden_imports, (
                                f"Forbidden import '{node.module}' detected in {path}"
                            )
