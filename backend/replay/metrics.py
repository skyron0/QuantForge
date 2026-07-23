from typing import Dict, Any, List
from decimal import Decimal
from backend.portfolio.models import PortfolioState, PortfolioSnapshot

def calculate_replay_metrics(
    initial_equity: float,
    final_state: PortfolioState,
    history: List[PortfolioSnapshot]
) -> Dict[str, Any]:
    """
    Computes key performance metrics from the simulated timeline.
    """
    final_equity = float(final_state.equity)
    realized_pnl = float(final_state.realized_pnl)
    unrealized_pnl = float(final_state.unrealized_pnl)
    total_fees = float(final_state.total_fees)

    # Gross return
    gross_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0

    # Drawdown tracking
    equity_values = [initial_equity] + [float(snap.equity) for snap in history]
    peak = initial_equity
    max_dd = 0.0

    for eq in equity_values:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_fees": total_fees,
        "gross_return": gross_return,
        "max_drawdown": max_dd,
        "min_equity": min(equity_values),
        "max_equity": max(equity_values)
    }
