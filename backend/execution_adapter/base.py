from abc import ABC, abstractmethod
from backend.execution_authorization.models import OrderIntent
from backend.execution_adapter.models import ExecutionResult, PaperExecutionContext

class BaseExecutionAdapter(ABC):
    """
    Abstract base class interface for execution adapters.
    Execution adapters consume already-authorized OrderIntent objects.
    They must not bypass or duplicate upstream authorization/sizing logic.
    """

    @abstractmethod
    def execute(self, intent: OrderIntent, context: PaperExecutionContext) -> ExecutionResult:
        """
        Executes the authorized order intent under the given market context constraints.
        Returns an ExecutionResult with fills, lineage details, and status.
        """
        pass
