from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from backend.risk.exceptions import InvalidRiskPolicyError


@dataclass(frozen=True)
class RiskPolicy:
    policy_version: str
    minimum_proposal_confidence: float
    maximum_proposal_age_seconds: float
    maximum_daily_loss_fraction: float  # e.g., 0.03 = 3% daily limit
    maximum_drawdown_fraction: float  # e.g., 0.05 = 5% drawdown limit
    maximum_portfolio_exposure_fraction: float  # e.g., 0.30 = 30% limit
    maximum_symbol_exposure_fraction: float  # e.g., 0.10 = 10% limit
    maximum_leverage: float
    maximum_open_positions: int
    maximum_symbol_open_positions: int
    maximum_consecutive_losses: int
    reject_on_critical_volatility: bool
    reject_on_critical_liquidity: bool
    reject_on_critical_drift: bool
    base_risk_fraction: float  # e.g., 0.01 = 1%
    maximum_risk_fraction: float
    minimum_risk_fraction: float
    blocking_risk_flags: Set[str] = field(default_factory=set)
    risk_reducing_flags: Dict[str, float] = field(default_factory=dict)
    informational_risk_flags: Set[str] = field(default_factory=set)
    volatility_adjustments: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Configuration checks
        if self.minimum_proposal_confidence < 0.0 or self.minimum_proposal_confidence > 1.0:
            raise InvalidRiskPolicyError(
                f"minimum_proposal_confidence ({self.minimum_proposal_confidence}) must be in [0.0, 1.0]"
            )
        if self.maximum_proposal_age_seconds <= 0.0:
            raise InvalidRiskPolicyError(
                f"maximum_proposal_age_seconds ({self.maximum_proposal_age_seconds}) must be greater than zero"
            )
        if self.maximum_daily_loss_fraction < 0.0 or self.maximum_daily_loss_fraction > 1.0:
            raise InvalidRiskPolicyError(
                f"maximum_daily_loss_fraction ({self.maximum_daily_loss_fraction}) must be in [0.0, 1.0]"
            )
        if self.maximum_drawdown_fraction < 0.0 or self.maximum_drawdown_fraction > 1.0:
            raise InvalidRiskPolicyError(
                f"maximum_drawdown_fraction ({self.maximum_drawdown_fraction}) must be in [0.0, 1.0]"
            )
        if (
            self.maximum_portfolio_exposure_fraction < 0.0
            or self.maximum_portfolio_exposure_fraction > 1.0
        ):
            raise InvalidRiskPolicyError(
                f"maximum_portfolio_exposure_fraction ({self.maximum_portfolio_exposure_fraction}) must be in [0.0, 1.0]"
            )
        if (
            self.maximum_symbol_exposure_fraction < 0.0
            or self.maximum_symbol_exposure_fraction > 1.0
        ):
            raise InvalidRiskPolicyError(
                f"maximum_symbol_exposure_fraction ({self.maximum_symbol_exposure_fraction}) must be in [0.0, 1.0]"
            )
        if self.maximum_leverage <= 0.0:
            raise InvalidRiskPolicyError(
                f"maximum_leverage ({self.maximum_leverage}) must be greater than zero"
            )
        if self.maximum_open_positions < 0:
            raise InvalidRiskPolicyError(
                f"maximum_open_positions ({self.maximum_open_positions}) cannot be negative"
            )
        if self.maximum_symbol_open_positions < 0:
            raise InvalidRiskPolicyError(
                f"maximum_symbol_open_positions ({self.maximum_symbol_open_positions}) cannot be negative"
            )
        if self.maximum_consecutive_losses < 0:
            raise InvalidRiskPolicyError(
                f"maximum_consecutive_losses ({self.maximum_consecutive_losses}) cannot be negative"
            )
        if self.base_risk_fraction < 0.0 or self.base_risk_fraction > 1.0:
            raise InvalidRiskPolicyError(
                f"base_risk_fraction ({self.base_risk_fraction}) must be in [0.0, 1.0]"
            )
        if self.minimum_risk_fraction < 0.0 or self.minimum_risk_fraction > 1.0:
            raise InvalidRiskPolicyError(
                f"minimum_risk_fraction ({self.minimum_risk_fraction}) must be in [0.0, 1.0]"
            )
        if self.maximum_risk_fraction < 0.0 or self.maximum_risk_fraction > 1.0:
            raise InvalidRiskPolicyError(
                f"maximum_risk_fraction ({self.maximum_risk_fraction}) must be in [0.0, 1.0]"
            )
        if self.minimum_risk_fraction > self.maximum_risk_fraction:
            raise InvalidRiskPolicyError(
                f"minimum_risk_fraction ({self.minimum_risk_fraction}) cannot exceed maximum_risk_fraction ({self.maximum_risk_fraction})"
            )
        if (
            self.base_risk_fraction < self.minimum_risk_fraction
            or self.base_risk_fraction > self.maximum_risk_fraction
        ):
            raise InvalidRiskPolicyError(
                f"base_risk_fraction ({self.base_risk_fraction}) must be within minimum ({self.minimum_risk_fraction}) and maximum ({self.maximum_risk_fraction}) limits"
            )

        # Flag and volatility adjustments validations
        for name, mult in self.risk_reducing_flags.items():
            if mult < 0.0 or mult > 1.0:
                raise InvalidRiskPolicyError(
                    f"risk_reducing_flags multiplier for '{name}' ({mult}) must be in [0.0, 1.0]"
                )
        for name, mult in self.volatility_adjustments.items():
            if mult < 0.0 or mult > 1.0:
                raise InvalidRiskPolicyError(
                    f"volatility_adjustments multiplier for '{name}' ({mult}) must be in [0.0, 1.0]"
                )
