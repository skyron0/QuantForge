from decimal import Decimal
from typing import Optional

from backend.risk.models import RiskContext
from backend.portfolio.models import PortfolioState

class PortfolioRiskContextBuilder:
    @staticmethod
    def build_risk_context(
        state: PortfolioState,
        symbol: str,
        volatility_state: str = "UNKNOWN",
        market_liquidity_state: str = "UNKNOWN",
        consecutive_losses: int = 0,
        drawdown_fraction: Decimal = Decimal("0"),
        metadata: Optional[dict] = None
    ) -> RiskContext:
        """
        Derives a RiskContext from current PortfolioState to serve as input for RiskGuardEngine.
        This bridge maps Decimal values to floats expected by RiskContext.
        """
        equity_float = float(state.equity)
        if equity_float <= 0.0:
            raise ValueError(f"Cannot generate RiskContext with non-positive equity: {equity_float}")
            
        pos = state.positions.get(symbol)
        symbol_exp_pct = 0.0
        symbol_pos_count = 0
        if pos:
            symbol_exp_pct = float(pos.position_notional) / equity_float
            symbol_pos_count = 1
            
        portfolio_exp_pct = float(state.gross_exposure) / equity_float
        current_leverage = float(state.gross_exposure) / equity_float
        
        return RiskContext(
            symbol=symbol,
            timestamp=state.timestamp,
            equity=equity_float,
            available_balance=float(state.available_balance),
            daily_realized_pnl=float(state.realized_pnl),
            daily_unrealized_pnl=float(state.unrealized_pnl),
            current_drawdown_pct=float(drawdown_fraction),
            portfolio_exposure_pct=portfolio_exp_pct,
            symbol_exposure_pct=symbol_exp_pct,
            current_leverage=current_leverage,
            open_positions_count=state.open_position_count,
            symbol_open_positions_count=symbol_pos_count,
            volatility_state=volatility_state,
            consecutive_losses=consecutive_losses,
            market_liquidity_state=market_liquidity_state,
            metadata=metadata or {}
        )
