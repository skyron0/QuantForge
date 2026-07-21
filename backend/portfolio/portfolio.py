import dataclasses
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
import threading

from backend.execution_authorization.models import OrderDirection
from backend.execution_adapter.models import ExecutionResult, Fill
from backend.portfolio.exceptions import (
    PortfolioError,
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
    Position,
    PositionSide,
    PortfolioState,
    PortfolioSnapshot
)
from backend.portfolio.policy import PortfolioPolicy
from backend.portfolio.idempotency import FillIdempotencyStore
from backend.portfolio.telemetry import PortfolioTelemetrySink

def to_decimal(val) -> Decimal:
    if val is None:
        return Decimal("0")
    if isinstance(val, Decimal):
        return val
    # Convert float to string first to avoid binary representation issues
    return Decimal(str(val))

class PortfolioEngine:
    def __init__(
        self,
        portfolio_id: str,
        initial_balance: Decimal,
        policy: PortfolioPolicy,
        idempotency_store: FillIdempotencyStore,
        telemetry_sink: Optional[PortfolioTelemetrySink] = None
    ):
        self.portfolio_id = portfolio_id
        self.initial_balance = to_decimal(initial_balance)
        self.policy = policy
        self.idempotency_store = idempotency_store
        self.telemetry_sink = telemetry_sink
        
        # Validations
        if self.initial_balance <= Decimal("0"):
            raise PortfolioValidationError("Initial balance must be positive")
            
        self._lock = threading.Lock()
        
        # Internal state variables
        self._cash_balance = self.initial_balance
        self._realized_pnl_acc = Decimal("0")
        self._total_fees = Decimal("0")
        
        # Maps symbol -> Position
        self._positions: Dict[str, Position] = {}
        
        # Metrics calculated mark-to-market
        self._equity = self.initial_balance
        self._unrealized_pnl = Decimal("0")
        self._used_margin = Decimal("0")
        self._available_balance = self.initial_balance
        self._gross_exposure = Decimal("0")
        self._net_exposure = Decimal("0")
        self._open_position_count = 0
        self._high_water_mark = self.initial_balance
        self._timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _validate_timestamp_unlocked(self, timestamp_str: str):
        try:
            ts_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception as e:
            raise PortfolioValidationError(f"Invalid timestamp format: {timestamp_str}") from e
            
        now_dt = datetime.now(timezone.utc)
        age = (now_dt - ts_dt).total_seconds()
        
        if age > self.policy.market_price_max_age_seconds:
            raise PortfolioValidationError(
                f"Clock skew error: timestamp is stale by {age:.2f} seconds (max {self.policy.market_price_max_age_seconds}s)"
            )
        if age < -self.policy.maximum_future_clock_skew_seconds:
            raise PortfolioValidationError(
                f"Clock skew error: timestamp is {abs(age):.2f}s in future (max {self.policy.maximum_future_clock_skew_seconds}s)"
            )

    def _recalculate_portfolio_unlocked(self, timestamp: str):
        total_unrealized = Decimal("0")
        total_margin = Decimal("0")
        gross_exp = Decimal("0")
        net_exp = Decimal("0")
        
        for pos in self._positions.values():
            total_unrealized += pos.unrealized_pnl
            total_margin += pos.margin_used
            gross_exp += pos.position_notional
            
            if pos.side == PositionSide.LONG:
                net_exp += pos.position_notional
            else:
                net_exp -= pos.position_notional
                
        # Main accounting equations
        # cash_balance is initial_balance + realized_pnl_acc - total_fees
        # equity = cash_balance + unrealized_pnl
        self._equity = self._cash_balance + total_unrealized
        self._unrealized_pnl = total_unrealized
        self._used_margin = total_margin
        self._gross_exposure = gross_exp
        self._net_exposure = net_exp
        self._available_balance = self._equity - total_margin
        self._open_position_count = len(self._positions)
        self._timestamp = timestamp
        
        if self._equity > self._high_water_mark:
            self._high_water_mark = self._equity

        # Invariants enforcement
        if self._gross_exposure < Decimal("0"):
            raise PortfolioInvariantError(f"Gross exposure cannot be negative: {self._gross_exposure}")
        if abs(self._net_exposure) > self._gross_exposure:
            raise PortfolioInvariantError(
                f"Gross exposure ({self._gross_exposure}) must be >= absolute net exposure ({abs(self._net_exposure)})"
            )
        if self._used_margin < Decimal("0"):
            raise PortfolioInvariantError(f"Used margin cannot be negative: {self._used_margin}")
        if self._total_fees < Decimal("0"):
            raise PortfolioInvariantError(f"Total fees cannot be negative: {self._total_fees}")

    def apply_execution_result(self, result: ExecutionResult) -> PortfolioState:
        """
        Atomically applies an ExecutionResult to update portfolio and position state.
        Guarantees transactional all-or-nothing (zero modification if any validation fails).
        """
        start_time = time.perf_counter()
        
        if not isinstance(result, ExecutionResult):
            raise TypeError("Expected ExecutionResult object")
            
        with self._lock:
            # 1. Total fee consistency check
            res_total_fees = to_decimal(result.total_fees)
            sum_fill_fees = sum(to_decimal(f.fee) for f in result.fills)
            if abs(res_total_fees - sum_fill_fees) > self.policy.accounting_tolerance:
                raise PortfolioInvariantError(
                    f"ExecutionResult total fee ({res_total_fees}) does not match fills sum ({sum_fill_fees})"
                )
                
            # 2. Check Idempotency
            fill_statuses = [self.idempotency_store.is_processed(f.fill_id) for f in result.fills]
            if len(fill_statuses) > 0 and all(fill_statuses):
                # Entire batch is already processed - return current state idempotently
                return self._get_state_unlocked()
            if any(fill_statuses):
                # Mixed new/duplicate state is invalid
                raise DuplicateFillError("ExecutionResult contains duplicate fills mixed with new fills")

            # 3. Time skew and status checks
            self._validate_timestamp_unlocked(result.completed_at)
            
            # Prepare draft state
            draft_positions = dict(self._positions)
            draft_cash_balance = self._cash_balance
            draft_realized_pnl_acc = self._realized_pnl_acc
            draft_total_fees = self._total_fees
            
            # 4. Ingest fills sequentially into draft state
            for fill in result.fills:
                # Basic validation
                fill_price = to_decimal(fill.price)
                fill_qty = to_decimal(fill.quantity)
                fill_fee = to_decimal(fill.fee)
                
                if fill_price.is_nan() or fill_price.is_infinite() or fill_price <= Decimal("0"):
                    raise InvalidFillError(f"Invalid fill price: {fill_price}")
                if fill_qty.is_nan() or fill_qty.is_infinite() or fill_qty <= Decimal("0"):
                    raise InvalidFillError(f"Invalid fill quantity: {fill_qty}")
                if fill_fee.is_nan() or fill_fee.is_infinite() or fill_fee < Decimal("0"):
                    raise InvalidFillError(f"Invalid fill fee: {fill_fee}")
                    
                # Supported instrument verification
                inst_type = fill.metadata.get("instrument_type", "linear_perpetual")
                if inst_type not in self.policy.supported_instrument_types:
                    raise UnsupportedInstrumentError(
                        f"Unsupported instrument type '{inst_type}' (matches policy limits)"
                    )
                    
                # Load leverage
                leverage = to_decimal(result.metadata.get("leverage", Decimal("1")))
                if leverage <= Decimal("0") or leverage > self.policy.maximum_leverage:
                    raise PortfolioValidationError(f"Invalid leverage parameter: {leverage}")
                    
                symbol = fill.symbol
                direction = fill.direction
                
                # Deduct fees from cash balance
                draft_cash_balance -= fill_fee
                draft_total_fees += fill_fee
                
                existing_pos = draft_positions.get(symbol)
                
                if not existing_pos:
                    # OPEN POSITION
                    side = PositionSide.LONG if direction == OrderDirection.BUY else PositionSide.SHORT
                    position_notional = fill_qty * fill_price
                    margin_used = position_notional / leverage
                    
                    new_pos = Position(
                        position_id=f"pos-{fill.fill_id}",
                        symbol=symbol,
                        side=side,
                        quantity=fill_qty,
                        average_entry_price=fill_price,
                        current_price=fill_price,
                        position_notional=position_notional,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=Decimal("0"),
                        accumulated_fees=fill_fee,
                        leverage=leverage,
                        margin_used=margin_used,
                        opened_at=fill.timestamp,
                        updated_at=fill.timestamp,
                        source_execution_ids=[result.execution_id],
                        source_fill_ids=[fill.fill_id],
                        metadata=dict(fill.metadata)
                    )
                    draft_positions[symbol] = new_pos
                    
                else:
                    # Existing position modified
                    pos_qty = existing_pos.quantity
                    pos_entry = existing_pos.average_entry_price
                    pos_side = existing_pos.side
                    
                    is_same_direction = (
                        (pos_side == PositionSide.LONG and direction == OrderDirection.BUY) or
                        (pos_side == PositionSide.SHORT and direction == OrderDirection.SELL)
                    )
                    
                    if is_same_direction:
                        # ADD TO POSITION
                        new_qty = pos_qty + fill_qty
                        new_entry = ((pos_qty * pos_entry) + (fill_qty * fill_price)) / new_qty
                        position_notional = new_qty * fill_price
                        
                        # Calculate unrealized pnl based on same-side formula
                        if pos_side == PositionSide.LONG:
                            unrealized = (fill_price - new_entry) * new_qty
                        else:
                            unrealized = (new_entry - fill_price) * new_qty
                            
                        margin_used = position_notional / leverage
                        
                        updated_pos = Position(
                            position_id=existing_pos.position_id,
                            symbol=symbol,
                            side=pos_side,
                            quantity=new_qty,
                            average_entry_price=new_entry,
                            current_price=fill_price,
                            position_notional=position_notional,
                            unrealized_pnl=unrealized,
                            realized_pnl=existing_pos.realized_pnl,
                            accumulated_fees=existing_pos.accumulated_fees + fill_fee,
                            leverage=leverage,
                            margin_used=margin_used,
                            opened_at=existing_pos.opened_at,
                            updated_at=fill.timestamp,
                            source_execution_ids=existing_pos.source_execution_ids + ([result.execution_id] if result.execution_id not in existing_pos.source_execution_ids else []),
                            source_fill_ids=existing_pos.source_fill_ids + ([fill.fill_id] if fill.fill_id not in existing_pos.source_fill_ids else []),
                            metadata=dict(existing_pos.metadata)
                        )
                        draft_positions[symbol] = updated_pos
                        
                    else:
                        # CLOSE OR REVERSAL
                        if fill_qty < pos_qty:
                            # PARTIAL CLOSE
                            closed_qty = fill_qty
                            remaining_qty = pos_qty - fill_qty
                            
                            if pos_side == PositionSide.LONG:
                                pnl = (fill_price - pos_entry) * closed_qty
                                unrealized = (fill_price - pos_entry) * remaining_qty
                            else:
                                pnl = (pos_entry - fill_price) * closed_qty
                                unrealized = (pos_entry - fill_price) * remaining_qty
                                
                            draft_realized_pnl_acc += pnl
                            draft_cash_balance += pnl
                            
                            position_notional = remaining_qty * fill_price
                            margin_used = position_notional / leverage
                            
                            updated_pos = Position(
                                position_id=existing_pos.position_id,
                                symbol=symbol,
                                side=pos_side,
                                quantity=remaining_qty,
                                average_entry_price=pos_entry, # preservations
                                current_price=fill_price,
                                position_notional=position_notional,
                                unrealized_pnl=unrealized,
                                realized_pnl=existing_pos.realized_pnl + pnl,
                                accumulated_fees=existing_pos.accumulated_fees + fill_fee,
                                leverage=leverage,
                                margin_used=margin_used,
                                opened_at=existing_pos.opened_at,
                                updated_at=fill.timestamp,
                                source_execution_ids=existing_pos.source_execution_ids + ([result.execution_id] if result.execution_id not in existing_pos.source_execution_ids else []),
                                source_fill_ids=existing_pos.source_fill_ids + ([fill.fill_id] if fill.fill_id not in existing_pos.source_fill_ids else []),
                                metadata=dict(existing_pos.metadata)
                            )
                            draft_positions[symbol] = updated_pos
                            
                        elif fill_qty == pos_qty:
                            # FULL CLOSE
                            closed_qty = pos_qty
                            if pos_side == PositionSide.LONG:
                                pnl = (fill_price - pos_entry) * closed_qty
                            else:
                                pnl = (pos_entry - fill_price) * closed_qty
                                
                            draft_realized_pnl_acc += pnl
                            draft_cash_balance += pnl
                            
                            # Remove position
                            del draft_positions[symbol]
                            
                        else:
                            # POSITION REVERSAL (fill_qty > pos_qty)
                            if not self.policy.allow_position_reversal:
                                raise InvalidPositionTransitionError(
                                    f"Position reversal not allowed by policy for {symbol}"
                                )
                                
                            closed_qty = pos_qty
                            remaining_qty = fill_qty - pos_qty
                            
                            if pos_side == PositionSide.LONG:
                                pnl = (fill_price - pos_entry) * closed_qty
                                reverse_side = PositionSide.SHORT
                            else:
                                pnl = (pos_entry - fill_price) * closed_qty
                                reverse_side = PositionSide.LONG
                                
                            draft_realized_pnl_acc += pnl
                            draft_cash_balance += pnl
                            
                            position_notional = remaining_qty * fill_price
                            margin_used = position_notional / leverage
                            
                            reversed_pos = Position(
                                position_id=f"pos-{fill.fill_id}",
                                symbol=symbol,
                                side=reverse_side,
                                quantity=remaining_qty,
                                average_entry_price=fill_price,
                                current_price=fill_price,
                                position_notional=position_notional,
                                unrealized_pnl=Decimal("0"),
                                realized_pnl=Decimal("0"),
                                accumulated_fees=fill_fee,
                                leverage=leverage,
                                margin_used=margin_used,
                                opened_at=fill.timestamp,
                                updated_at=fill.timestamp,
                                source_execution_ids=[result.execution_id],
                                source_fill_ids=[fill.fill_id],
                                metadata=dict(fill.metadata)
                            )
                            draft_positions[symbol] = reversed_pos
            
            # Temporary state check helper for policy verification
            # Recalculate totals on temporary workspace to verify invariants
            tot_unrealized = Decimal("0")
            tot_margin = Decimal("0")
            g_exp = Decimal("0")
            n_exp = Decimal("0")
            for pos in draft_positions.values():
                tot_unrealized += pos.unrealized_pnl
                tot_margin += pos.margin_used
                g_exp += pos.position_notional
                if pos.side == PositionSide.LONG:
                    n_exp += pos.position_notional
                else:
                    n_exp -= pos.position_notional
                    
            tmp_equity = draft_cash_balance + tot_unrealized
            
            # Exposure limits verification
            if tmp_equity > Decimal("0"):
                g_exp_frac = g_exp / tmp_equity
                n_exp_frac = abs(n_exp) / tmp_equity
                if g_exp_frac > self.policy.maximum_gross_exposure_fraction:
                    raise PortfolioInvariantError(
                        f"Gross exposure fraction ({g_exp_frac}) exceeds limit of {self.policy.maximum_gross_exposure_fraction}"
                    )
                if n_exp_frac > self.policy.maximum_net_exposure_fraction:
                    raise PortfolioInvariantError(
                        f"Net exposure fraction ({n_exp_frac}) exceeds limit of {self.policy.maximum_net_exposure_fraction}"
                    )
                    
            # Open position bounds check
            if len(draft_positions) > self.policy.maximum_open_positions:
                raise PortfolioInvariantError(
                    f"Open positions count ({len(draft_positions)}) exceeds limit of {self.policy.maximum_open_positions}"
                )
                
            # If everything succeeded, commit!
            self._positions = draft_positions
            self._cash_balance = draft_cash_balance
            self._realized_pnl_acc = draft_realized_pnl_acc
            self._total_fees = draft_total_fees
            self._recalculate_portfolio_unlocked(result.completed_at)
            
            # Record processed fills
            for fill in result.fills:
                self.idempotency_store.record(fill.fill_id)
                
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Telemetry sink notification
            active_state = self._get_state_unlocked()
            if self.telemetry_sink:
                self.telemetry_sink.record_update(active_state, latency_ms, "SUCCESS")
                
            return active_state

    def update_market_price(self, symbol: str, price: float, timestamp: str) -> PortfolioState:
        """
        Updates the current market mark price of a specific symbol and recalculates unrealized PnL/exposures.
        Does NOT build ExecutionResults or Fills.
        """
        start_time = time.perf_counter()
        
        price_dec = to_decimal(price)
        if price_dec.is_nan() or price_dec.is_infinite() or price_dec <= Decimal("0"):
            raise PortfolioValidationError(f"Invalid market price update: {price}")
            
        with self._lock:
            self._validate_timestamp_unlocked(timestamp)
            
            pos = self._positions.get(symbol)
            if pos:
                # Update position mark pricing page
                pos_qty = pos.quantity
                pos_entry = pos.average_entry_price
                position_notional = pos_qty * price_dec
                
                # Calculations
                if pos.side == PositionSide.LONG:
                    unrealized = (price_dec - pos_entry) * pos_qty
                else:
                    unrealized = (pos_entry - price_dec) * pos_qty
                    
                margin_used = position_notional / pos.leverage
                
                updated_pos = dataclasses.replace(
                    pos,
                    current_price=price_dec,
                    position_notional=position_notional,
                    unrealized_pnl=unrealized,
                    margin_used=margin_used,
                    updated_at=timestamp
                )
                self._positions[symbol] = updated_pos
                
            self._recalculate_portfolio_unlocked(timestamp)
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            active_state = self._get_state_unlocked()
            if self.telemetry_sink:
                self.telemetry_sink.record_update(active_state, latency_ms, "MARK_PRICE_UPDATE")
                
            return active_state

    def get_position(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol)

    def get_positions(self) -> List[Position]:
        with self._lock:
            return list(self._positions.values())

    def _get_state_unlocked(self) -> PortfolioState:
        return PortfolioState(
            portfolio_id=self.portfolio_id,
            initial_balance=self.initial_balance,
            cash_balance=self._cash_balance,
            equity=self._equity,
            realized_pnl=self._realized_pnl_acc,
            unrealized_pnl=self._unrealized_pnl,
            total_fees=self._total_fees,
            used_margin=self._used_margin,
            available_balance=self._available_balance,
            gross_exposure=self._gross_exposure,
            net_exposure=self._net_exposure,
            open_position_count=self._open_position_count,
            positions=dict(self._positions),
            timestamp=self._timestamp,
            metadata={"high_water_mark": float(self._high_water_mark)}
        )

    def get_state(self) -> PortfolioState:
        with self._lock:
            return self._get_state_unlocked()

    def create_snapshot(self) -> PortfolioSnapshot:
        with self._lock:
            return PortfolioSnapshot(
                portfolio_id=self.portfolio_id,
                initial_balance=self.initial_balance,
                cash_balance=self._cash_balance,
                equity=self._equity,
                realized_pnl=self._realized_pnl_acc,
                unrealized_pnl=self._unrealized_pnl,
                total_fees=self._total_fees,
                used_margin=self._used_margin,
                available_balance=self._available_balance,
                gross_exposure=self._gross_exposure,
                net_exposure=self._net_exposure,
                open_positions=list(self._positions.values()),
                timestamp=self._timestamp,
                metadata={"high_water_mark": float(self._high_water_mark)}
            )

    @property
    def high_water_mark(self) -> Decimal:
        with self._lock:
            return self._high_water_mark

    @property
    def drawdown_fraction(self) -> Decimal:
        with self._lock:
            if self._high_water_mark <= Decimal("0"):
                return Decimal("0")
            drawdown = (self._high_water_mark - self._equity) / self._high_water_mark
            # Ensure it is bounded at >= 0
            return max(Decimal("0"), drawdown)

    def clear(self):
        with self._lock:
            self._positions.clear()
            self._cash_balance = self.initial_balance
            self._realized_pnl_acc = Decimal("0")
            self._total_fees = Decimal("0")
            self._equity = self.initial_balance
            self._unrealized_pnl = Decimal("0")
            self._used_margin = Decimal("0")
            self._available_balance = self.initial_balance
            self._gross_exposure = Decimal("0")
            self._net_exposure = Decimal("0")
            self._open_position_count = 0
            self._high_water_mark = self.initial_balance
            self._timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self.idempotency_store.clear()

class Portfolio:
    START_BALANCE = 10000.0

    def __init__(self):
        self.balance = self.START_BALANCE
        self.cash = self.START_BALANCE
        self.positions = []
        self.total_profit = 0.0

    def open_position(self, position):
        self.positions.append(position)

    def close_position(self, position):
        if position not in self.positions:
            return
        position.is_open = False
        self.positions.remove(position)
        self.total_profit += position.pnl

    def get_open_positions(self):
        return [p for p in self.positions if p.is_open]

    def has_open_position(self, symbol):
        return any(p.symbol == symbol and p.is_open for p in self.positions)

    def update_positions(self, candle):
        for position in self.get_open_positions():
            if position.symbol != candle.symbol:
                continue
            if position.side == "BUY":
                position.pnl = (candle.close - position.entry_price) * position.quantity

    def equity(self):
        return self.balance + self.total_profit