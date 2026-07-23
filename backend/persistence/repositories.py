from abc import ABC, abstractmethod
from typing import List, Optional
from backend.persistence.models import (
    RuntimeSessionRecord, TradingCycleRecord, DecisionRecord,
    TradeProposalRecord, RiskAuthorizationRecord, PositionSizingRecord,
    OrderIntentRecord, ExecutionRecord, FillRecord,
    PortfolioSnapshotRecord, PositionLifecycleRecord, AuditEventRecord
)


class RuntimeSessionRepository(ABC):
    @abstractmethod
    def save(self, record: RuntimeSessionRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, session_id: str) -> Optional[RuntimeSessionRecord]:
        pass


class TradingCycleRepository(ABC):
    @abstractmethod
    def save(self, record: TradingCycleRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, cycle_id: str) -> Optional[TradingCycleRecord]:
        pass

    @abstractmethod
    def list_by_session(self, session_id: str) -> List[TradingCycleRecord]:
        pass


class DecisionRepository(ABC):
    @abstractmethod
    def save(self, record: DecisionRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, decision_id: str) -> Optional[DecisionRecord]:
        pass

    @abstractmethod
    def list_by_cycle(self, cycle_id: str) -> List[DecisionRecord]:
        pass


class TradeProposalRepository(ABC):
    @abstractmethod
    def save(self, record: TradeProposalRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, proposal_id: str) -> Optional[TradeProposalRecord]:
        pass


class RiskAuthorizationRepository(ABC):
    @abstractmethod
    def save(self, record: RiskAuthorizationRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, authorization_id: str) -> Optional[RiskAuthorizationRecord]:
        pass


class PositionSizingRepository(ABC):
    @abstractmethod
    def save(self, record: PositionSizingRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, sizing_id: str) -> Optional[PositionSizingRecord]:
        pass


class OrderIntentRepository(ABC):
    @abstractmethod
    def save(self, record: OrderIntentRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, intent_id: str) -> Optional[OrderIntentRecord]:
        pass


class ExecutionRepository(ABC):
    @abstractmethod
    def save(self, record: ExecutionRecord, fills: List[FillRecord]) -> None:
        pass

    @abstractmethod
    def get_by_id(self, execution_id: str) -> Optional[ExecutionRecord]:
        pass

    @abstractmethod
    def get_fills(self, execution_id: str) -> List[FillRecord]:
        pass


class PortfolioRepository(ABC):
    @abstractmethod
    def save_snapshot(self, snapshot: PortfolioSnapshotRecord) -> None:
        pass

    @abstractmethod
    def get_snapshot(self, portfolio_snapshot_id: str) -> Optional[PortfolioSnapshotRecord]:
        pass

    @abstractmethod
    def list_snapshots_by_portfolio(self, portfolio_id: str) -> List[PortfolioSnapshotRecord]:
        pass


class PositionLifecycleRepository(ABC):
    @abstractmethod
    def save(self, record: PositionLifecycleRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, lifecycle_id: str) -> Optional[PositionLifecycleRecord]:
        pass

    @abstractmethod
    def list_by_symbol(self, symbol: str) -> List[PositionLifecycleRecord]:
        pass


class AuditRepository(ABC):
    @abstractmethod
    def save(self, record: AuditEventRecord) -> None:
        pass

    @abstractmethod
    def get_by_id(self, audit_id: str) -> Optional[AuditEventRecord]:
        pass

    @abstractmethod
    def get_latest_event(self, session_id: str) -> Optional[AuditEventRecord]:
        pass

    @abstractmethod
    def get_chain(self, session_id: str) -> List[AuditEventRecord]:
        pass
