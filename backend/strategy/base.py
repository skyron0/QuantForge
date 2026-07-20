from abc import ABC
from abc import abstractmethod
from backend.decision.models import Decision


class BaseStrategy(ABC):

    @abstractmethod
    def decide(self, features) -> Decision | None:
        """
        Evaluate features and generate a BUY, SELL, or HOLD decision.
        """
        pass
