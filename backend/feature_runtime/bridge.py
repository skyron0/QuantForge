"""
EventBus bridge coupling MarketDataSnapshotUpdated events to the
FeatureRuntimeService, then publishing MLSignalGenerated events.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from backend.runtime.events import TradingEvent
from backend.runtime.event_bus import BaseEventBus
from backend.feature_runtime.service import FeatureRuntimeService
from backend.feature_runtime.models import FeatureRuntimeStatus
from backend.decision.models import MLSignal


@dataclass(frozen=True)
class MLSignalGenerated(TradingEvent):
    """Event emitted when the Feature Runtime successfully produces an MLSignal."""
    ml_signal: Optional[MLSignal] = None


class FeatureRuntimeBridge:
    """
    Subscribes to MarketDataSnapshotUpdated events and triggers the
    Feature Runtime pipeline.  On success, publishes an MLSignalGenerated
    event to the EventBus.
    """

    def __init__(
        self,
        event_bus: BaseEventBus,
        service: FeatureRuntimeService,
        symbols: Optional[list] = None,
        runtime_id: str = "feature-bridge",
        session_id: str = "feature-session",
    ) -> None:
        self.event_bus = event_bus
        self.service = service
        self.symbols = symbols  # None = process all symbols
        self.runtime_id = runtime_id
        self.session_id = session_id

    def on_snapshot(self, symbol: str, timestamp: str) -> Optional[MLSignal]:
        """
        Called when a new market data snapshot is available.

        Returns the generated MLSignal on success, or None on skip/failure.
        """
        if self.symbols is not None and symbol not in self.symbols:
            return None

        result = self.service.process(symbol, timestamp)

        if result.status == FeatureRuntimeStatus.SUCCESS and result.ml_signal:
            event = MLSignalGenerated(
                event_id=str(uuid.uuid4()),
                event_type="MLSignalGenerated",
                timestamp=datetime.now(timezone.utc).isoformat(),
                runtime_id=self.runtime_id,
                session_id=self.session_id,
                cycle_id=None,
                ml_signal=result.ml_signal,
            )
            self.event_bus.publish(event)
            return result.ml_signal

        return None
