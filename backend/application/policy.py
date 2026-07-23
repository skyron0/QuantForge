from dataclasses import dataclass, field
from typing import List, Optional
from backend.application.exceptions import ApplicationConfigurationError


@dataclass(frozen=True)
class IntegratedRuntimePolicy:
    """
    Orchestration and session lifecycle configuration policy.
    Controls application layer coordination behavior only; does not duplicate domain limits.
    Enforces that execution MUST be paper-only.
    """
    policy_version: str
    paper_only: bool = True
    enabled_symbols: List[str] = field(default_factory=list)
    enabled_timeframes: List[str] = field(default_factory=list)
    cycle_trigger_mode: str = "CANDLE_CLOSE"  # "CANDLE_CLOSE", "REPLAY_STEP", "MANUAL"
    
    max_cycles_per_session: Optional[int] = None
    max_session_duration_seconds: Optional[float] = None
    
    stop_on_portfolio_failure: bool = True
    stop_on_persistence_failure: bool = True
    stop_on_unhandled_exception: bool = True
    continue_after_cycle_rejection: bool = True
    
    require_persistence: bool = False
    require_market_data_health: bool = True
    require_model_health: bool = True
    
    warmup_cycles_allowed: int = 100
    max_consecutive_cycle_failures: int = 5
    health_check_interval_cycles: int = 10
    graceful_shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.paper_only:
            raise ApplicationConfigurationError(
                "CRITICAL: LIVE trading is explicitly disallowed. 'paper_only' must be set to True."
            )
        
        if not self.policy_version:
            raise ApplicationConfigurationError("policy_version must not be empty.")
            
        if not self.enabled_symbols:
            raise ApplicationConfigurationError("enabled_symbols list must contain at least one symbol.")
            
        if not self.enabled_timeframes:
            raise ApplicationConfigurationError("enabled_timeframes list must contain at least one timeframe.")

        valid_triggers = ("CANDLE_CLOSE", "REPLAY_STEP", "MANUAL", "MARKET_UPDATE")
        if self.cycle_trigger_mode not in valid_triggers:
            raise ApplicationConfigurationError(
                f"invalid cycle_trigger_mode: {self.cycle_trigger_mode}. Must be one of {valid_triggers}."
            )
            
        if self.warmup_cycles_allowed < 0:
            raise ApplicationConfigurationError("warmup_cycles_allowed must be non-negative.")

        if self.max_consecutive_cycle_failures <= 0:
            raise ApplicationConfigurationError("max_consecutive_cycle_failures must be positive.")
            
        if self.health_check_interval_cycles <= 0:
            raise ApplicationConfigurationError("health_check_interval_cycles must be positive.")
            
        if self.graceful_shutdown_timeout_seconds <= 0.0:
            raise ApplicationConfigurationError("graceful_shutdown_timeout_seconds must be positive.")
