from decimal import Decimal
from typing import List, Optional, Tuple, Dict
from backend.portfolio.models import PositionSide
from backend.position_lifecycle.exceptions import (
    InvalidStopLossError,
    InvalidTakeProfitError,
    InvalidTrailingStopError,
    ProtectiveLevelError
)
from backend.position_lifecycle.models import ProtectiveTriggerType, ProtectivePositionState
from backend.position_lifecycle.policy import PositionLifecyclePolicy


def validate_protective_levels(
    side: PositionSide,
    entry_price: Decimal,
    stop_loss: Optional[Decimal],
    take_profit: Optional[Decimal],
    trailing_stop_enabled: bool,
    trailing_distance: Optional[Decimal],
    trailing_activation_price: Optional[Decimal],
    policy: PositionLifecyclePolicy
) -> None:
    """
    Validates protective bounds and settings against policy rules and math constraints.
    """
    # Ensure no NaN/Inf/non-positive values are passed
    if entry_price.is_nan() or entry_price.is_infinite():
        raise ValueError("entry_price must not be NaN/Infinite")
    if entry_price <= Decimal("0"):
        raise ValueError("entry_price must be positive")
        
    for name, val in [
        ("stop_loss", stop_loss),
        ("take_profit", take_profit),
        ("trailing_distance", trailing_distance),
        ("trailing_activation_price", trailing_activation_price)
    ]:
        if val is not None:
            if val.is_nan() or val.is_infinite():
                raise ValueError(f"{name} must not be NaN/Infinite")
            if val <= Decimal("0"):
                raise ValueError(f"{name} must be positive")

    # 1. Stop Loss Validation
    if stop_loss is not None:
        if not policy.allow_stop_loss:
            raise InvalidStopLossError("Stop loss is not allowed by policy")
        
        # Check orientation
        if side == PositionSide.LONG:
            if stop_loss >= entry_price:
                raise InvalidStopLossError("LONG stop loss must be strictly below entry price")
        elif side == PositionSide.SHORT:
            if stop_loss <= entry_price:
                raise InvalidStopLossError("SHORT stop loss must be strictly above entry price")
        
        # Check distance bounds
        distance_pct = abs(entry_price - stop_loss) / entry_price
        if distance_pct < policy.minimum_stop_distance_fraction:
            raise InvalidStopLossError("Stop loss is closer than minimum allowed distance")
        if distance_pct > policy.maximum_stop_distance_fraction:
            raise InvalidStopLossError("Stop loss exceeds maximum allowed distance")
    else:
        if policy.require_stop_loss:
            raise InvalidStopLossError("Stop loss is required by policy but not provided")

    # 2. Take Profit Validation
    if take_profit is not None:
        if not policy.allow_take_profit:
            raise InvalidTakeProfitError("Take profit is not allowed by policy")
            
        # Check orientation
        if side == PositionSide.LONG:
            if take_profit <= entry_price:
                raise InvalidTakeProfitError("LONG take profit must be strictly above entry price")
        elif side == PositionSide.SHORT:
            if take_profit >= entry_price:
                raise InvalidTakeProfitError("SHORT take profit must be strictly below entry price")
                
        # Check distance bounds
        distance_pct = abs(take_profit - entry_price) / entry_price
        if distance_pct < policy.minimum_take_profit_distance_fraction:
            raise InvalidTakeProfitError("Take profit is closer than minimum allowed distance")
    else:
        if policy.require_take_profit:
            raise InvalidTakeProfitError("Take profit is required by policy but not provided")

    # 3. Trailing Stop Validation
    if trailing_stop_enabled:
        if not policy.allow_trailing_stop:
            raise InvalidTrailingStopError("Trailing stop is not allowed by policy")
        if trailing_distance is None:
            raise InvalidTrailingStopError("trailing_distance is required when trailing stop is enabled")
            
        # Validate trailing distance mode and limits
        if policy.trailing_distance_mode == "PERCENTAGE":
            # For percentage mode, distance is represented as fraction
            if trailing_distance < policy.minimum_trailing_distance or trailing_distance > policy.maximum_trailing_distance:
                raise InvalidTrailingStopError("Trailing distance fraction violates policy limits")
        else:  # ABSOLUTE
            if trailing_distance < policy.minimum_trailing_distance or trailing_distance > policy.maximum_trailing_distance:
                raise InvalidTrailingStopError("Trailing distance violates policy limits")
                
        if trailing_activation_price is not None:
            if side == PositionSide.LONG:
                if trailing_activation_price < entry_price:
                    raise InvalidTrailingStopError("LONG trailing activation price must be >= entry price")
            else:
                if trailing_activation_price > entry_price:
                    raise InvalidTrailingStopError("SHORT trailing activation price must be <= entry price")


def update_trailing_and_check_triggers(
    state: ProtectivePositionState,
    market_price: Decimal,
    policy: PositionLifecyclePolicy
) -> Tuple[Optional[ProtectiveTriggerType], Decimal, Optional[Decimal], Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
    """
    Updates trailing stop values based on market price and evaluates whether stop-loss,
    take-profit, or trailing stop levels are triggered.
    
    Returns:
        (triggered_type, trigger_price, new_highest, new_lowest, new_active_stop, new_activation_price)
    """
    if market_price <= Decimal("0") or market_price.is_nan() or market_price.is_infinite():
        raise ValueError(f"Invalid market price: {market_price}")

    # Track highs/lows
    new_highest = state.highest_price_since_entry
    new_lowest = state.lowest_price_since_entry
    if new_highest is None or market_price > new_highest:
        new_highest = market_price
    if new_lowest is None or market_price < new_lowest:
        new_lowest = market_price

    new_active_stop = state.active_trailing_stop_price
    
    # 1. Update Trailing Stop Price if enabled
    if state.trailing_stop_enabled:
        trailing_dist = state.trailing_distance
        if trailing_dist is None:
            raise InvalidTrailingStopError("trailing_distance is required when trailing stop is enabled")
            
        if policy.trailing_distance_mode == "PERCENTAGE":
            # trailing_distance is a fraction of the current highest/lowest price
            if state.side == PositionSide.LONG:
                calc_dist = new_highest * trailing_dist
            else:
                calc_dist = new_lowest * trailing_dist
        else:
            calc_dist = trailing_dist

        # Check activation
        is_activated = True
        if state.trailing_activation_price is not None:
            if state.side == PositionSide.LONG:
                # Active only if new_highest has touched/crossed activation price
                is_activated = new_highest >= state.trailing_activation_price
            else:
                is_activated = new_lowest <= state.trailing_activation_price

        if is_activated:
            if state.side == PositionSide.LONG:
                calc_stop = new_highest - calc_dist
                if new_active_stop is None or calc_stop > new_active_stop:
                    new_active_stop = calc_stop
            else:  # SHORT
                calc_stop = new_lowest + calc_dist
                if new_active_stop is None or calc_stop < new_active_stop:
                    new_active_stop = calc_stop

    # Breakeven Adjustment (if policy allows it and allowed breakeven is configured)
    # Moving STOP_LOSS to entry or offset after favorable movement
    new_stop_loss = state.stop_loss
    if policy.allow_breakeven and policy.breakeven_activation_fraction is not None:
        activation_move = state.average_entry_price * policy.breakeven_activation_fraction
        offset_move = state.average_entry_price * (policy.breakeven_offset_fraction or Decimal("0"))
        
        if state.side == PositionSide.LONG:
            # Activates if highest price has moved entry + activation distance
            if new_highest >= state.average_entry_price + activation_move:
                target_stop = state.average_entry_price + offset_move
                # Breakeven MUST only reduce risk, i.e., raise the LONG SL. Moving it down is forbidden.
                if new_stop_loss is None or target_stop > new_stop_loss:
                    # Capped at current price to avoid instant triggering
                    new_stop_loss = min(target_stop, market_price - Decimal("0.00000001"))
        else:  # SHORT
            if new_lowest <= state.average_entry_price - activation_move:
                target_stop = state.average_entry_price - offset_move
                # Breakeven MUST only reduce risk, i.e., lower the SHORT SL. Moving it up is forbidden.
                if new_stop_loss is None or target_stop < new_stop_loss:
                    new_stop_loss = max(target_stop, market_price + Decimal("0.00000001"))

    # 2. Check Triggers
    triggered: Dict[ProtectiveTriggerType, Decimal] = {}

    # Check hard Stop Loss
    if new_stop_loss is not None:
        if state.side == PositionSide.LONG:
            if market_price <= new_stop_loss:
                triggered[ProtectiveTriggerType.STOP_LOSS] = new_stop_loss
        else:  # SHORT
            if market_price >= new_stop_loss:
                triggered[ProtectiveTriggerType.STOP_LOSS] = new_stop_loss

    # Check Take Profit
    if state.take_profit is not None:
        if state.side == PositionSide.LONG:
            if market_price >= state.take_profit:
                triggered[ProtectiveTriggerType.TAKE_PROFIT] = state.take_profit
        else:  # SHORT
            if market_price <= state.take_profit:
                triggered[ProtectiveTriggerType.TAKE_PROFIT] = state.take_profit

    # Check Trailing Stop
    if state.trailing_stop_enabled and new_active_stop is not None:
        if state.side == PositionSide.LONG:
            if market_price <= new_active_stop:
                triggered[ProtectiveTriggerType.TRAILING_STOP] = new_active_stop
        else:  # SHORT
            if market_price >= new_active_stop:
                triggered[ProtectiveTriggerType.TRAILING_STOP] = new_active_stop

    # Determine priority trigger if multiple hit simultaneously (within the same observation)
    if triggered:
        for t_type in policy.trigger_priority:
            if t_type in triggered:
                return t_type, triggered[t_type], new_highest, new_lowest, new_active_stop, new_stop_loss

    return None, market_price, new_highest, new_lowest, new_active_stop, new_stop_loss
