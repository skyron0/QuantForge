from typing import List, Dict, Any, Callable
from backend.replay.clock import ReplayClock

class ReplayScheduler:
    """
    Step-driven, deterministic scheduler that sequences historical data records.
    Advances the ReplayClock to the timestamp of each step before invoking transaction handlers.
    """
    def __init__(
        self,
        clock: ReplayClock,
        dataset: List[Dict[str, Any]],
        on_step: Callable[[Dict[str, Any]], None]
    ) -> None:
        self.clock = clock
        self.dataset = dataset
        self.on_step = on_step
        self._index = 0

    @property
    def total_steps(self) -> int:
        return len(self.dataset)

    @property
    def current_index(self) -> int:
        return self._index

    def step(self) -> bool:
        """
        Executes a single step:
        1. Advances clock to current record timestamp.
        2. Invokes callback on_step.
        Returns:
            bool: True if step executed successfully, False if dataset exhausted.
        """
        if self._index >= len(self.dataset):
            return False

        record = self.dataset[self._index]
        ts_str = record.get("timestamp") or record.get("open_time")
        if ts_str:
            self.clock.set_time(ts_str)

        # Execute callback
        self.on_step(record)
        
        self._index += 1
        return True

    def run(self) -> None:
        """Runs the simulation to exhaustion."""
        while self.step():
            pass
