from dataclasses import dataclass
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.runtime.events import TradingEvent
from backend.runtime.event_bus import BaseEventBus
from backend.market_data.models import MarketDataEnvelope, MarketDataSnapshot
from backend.market_data.service import MarketDataService
from backend.market_data.snapshot import MarketDataSnapshotBuilder


@dataclass(frozen=True)
class MarketDataMessageReceived(TradingEvent):
    envelope: Optional[MarketDataEnvelope] = None


@dataclass(frozen=True)
class MarketDataSnapshotUpdated(TradingEvent):
    snapshot: Optional[MarketDataSnapshot] = None


class MarketDataBridge:
    """
    Publishing bridge mapping normalized envelopes and point-in-time snapshots
    directly to EventBus as immutable TradingEvent instances.
    """

    def __init__(
        self,
        event_bus: BaseEventBus,
        market_data_service: MarketDataService,
        snapshot_builder: MarketDataSnapshotBuilder,
        runtime_id: str = "bridge-runtime",
        session_id: str = "bridge-session"
    ) -> None:
        self.event_bus = event_bus
        self.market_data_service = market_data_service
        self.snapshot_builder = snapshot_builder
        self.runtime_id = runtime_id
        self.session_id = session_id

    def publish_envelope(self, envelope: MarketDataEnvelope) -> None:
        event = MarketDataMessageReceived(
            event_id=str(uuid.uuid4()),
            event_type="MarketDataMessageReceived",
            timestamp=datetime.now(timezone.utc).isoformat(),
            runtime_id=self.runtime_id,
            session_id=self.session_id,
            cycle_id=None,
            envelope=envelope
        )
        self.event_bus.publish(event)

    def publish_snapshot(self, snapshot: MarketDataSnapshot) -> None:
        event = MarketDataSnapshotUpdated(
            event_id=str(uuid.uuid4()),
            event_type="MarketDataSnapshotUpdated",
            timestamp=datetime.now(timezone.utc).isoformat(),
            runtime_id=self.runtime_id,
            session_id=self.session_id,
            cycle_id=None,
            snapshot=snapshot
        )
        self.event_bus.publish(event)
