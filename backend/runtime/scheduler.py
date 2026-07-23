import time
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any

from backend.orchestration.models import TradingCycleInput
from backend.runtime.exceptions import SchedulerError


class BaseScheduler(ABC):
    """Abstract interface defining the scheduler capability."""
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass

    @abstractmethod
    def resume(self) -> None:
        pass

    @abstractmethod
    def tick(self) -> None:
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_paused(self) -> bool:
        pass


class TradingCycleScheduler(BaseScheduler):
    """
    Polling coordinator responsible for periodic execution of the TradingCycleOrchestrator.
    Does not contain any trade/risk business logic. Can be ticked manually in tests.
    """
    def __init__(
        self,
        interval_seconds: float,
        input_provider: Callable[[], Optional[TradingCycleInput]],
        cycle_executor: Callable[[TradingCycleInput], Any],
        telemetry: Optional[Any] = None
    ) -> None:
        self.interval_seconds = interval_seconds
        self.input_provider = input_provider
        self.cycle_executor = cycle_executor
        self.telemetry = telemetry

        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the periodic scheduler thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._paused = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="TradingCycleSchedulerLoop",
                daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stops the scheduler and joins execution thread."""
        thread_to_join = None
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._paused = False
            thread_to_join = self._thread
            self._thread = None

        if thread_to_join and thread_to_join.is_alive():
            thread_to_join.join(timeout=2.0)

    def pause(self) -> None:
        """Pauses the scheduler periodic execution."""
        with self._lock:
            if not self._running:
                raise SchedulerError("Cannot pause a stopped/inactive scheduler")
            self._paused = True

    def resume(self) -> None:
        """Resumes the scheduler periodic execution."""
        with self._lock:
            if not self._running:
                raise SchedulerError("Cannot resume a stopped/inactive scheduler")
            self._paused = False

    def tick(self) -> None:
        """
        Executes a single step / iteration of the cycle scheduler.
        Designed for both manual test ticking and the main loop.
        """
        # Read running state thread-safely
        with self._lock:
            if not self._running or self._paused:
                return

        # Fetch cycle inputs using provider
        input_data = None
        try:
            input_data = self.input_provider()
        except Exception as e:
            if self.telemetry:
                self.telemetry.record_runtime_error(f"Input provider failed: {str(e)}")
            return

        if input_data is None:
            return

        start_time = time.perf_counter()
        
        # Execute cycle
        try:
            self.cycle_executor(input_data)
        except Exception as e:
            if self.telemetry:
                self.telemetry.record_runtime_error(f"Cycle execution failed: {str(e)}")
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            if self.telemetry:
                self.telemetry.increment_scheduler_iterations()
                self.telemetry.record_cycle(duration_ms)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def _run_loop(self) -> None:
        """Private loop executed by background polling thread."""
        while True:
            # Check exit conditions
            with self._lock:
                if not self._running:
                    break

            self.tick()
            
            # Sub-second sleep checks to allow quick shutdown
            sleep_step = min(0.1, self.interval_seconds)
            elapsed = 0.0
            while elapsed < self.interval_seconds:
                with self._lock:
                    if not self._running:
                        break
                time.sleep(sleep_step)
                elapsed += sleep_step
