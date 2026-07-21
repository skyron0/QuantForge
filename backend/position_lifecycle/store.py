import threading
from collections import OrderedDict
from typing import Dict, Optional
from backend.position_lifecycle.exceptions import (
    DuplicateTriggerError,
    PositionNotFoundError,
    PositionStateError
)
from backend.position_lifecycle.models import PositionLifecycleStatus, ProtectivePositionState


class PositionLifecycleStore:
    def __init__(self, max_closed_capacity: int = 1000):
        self._lock = threading.RLock()
        # position_id -> ProtectivePositionState for active (OPEN or CLOSING) positions
        self._active_states: Dict[str, ProtectivePositionState] = {}
        # Bounded store for CLOSED positions to preserve audit state and prevent memory leaks.
        # list of position_id -> ProtectivePositionState
        self._closed_states: OrderedDict[str, ProtectivePositionState] = OrderedDict()
        self._max_closed_capacity = max_closed_capacity

    def put(self, state: ProtectivePositionState) -> None:
        """
        Adds a new active ProtectivePositionState to the store.
        Ensures there is no active lifecycle already registered for this position.
        """
        with self._lock:
            pos_id = state.position_id
            if pos_id in self._active_states:
                raise PositionStateError(
                    f"An active lifecycle already exists for position {pos_id}"
                )
            if state.status == PositionLifecycleStatus.CLOSED:
                self._add_to_closed(state)
            else:
                self._active_states[pos_id] = state

    def get(self, position_id: str) -> Optional[ProtectivePositionState]:
        """
        Retrieves a state by position_id. Looks in active first, then closed.
        """
        with self._lock:
            if position_id in self._active_states:
                return self._active_states[position_id]
            return self._closed_states.get(position_id)

    def get_by_lifecycle_id(self, lifecycle_id: str) -> Optional[ProtectivePositionState]:
        """
        Retrieves a state by lifecycle_id.
        """
        with self._lock:
            for state in self._active_states.values():
                if state.lifecycle_id == lifecycle_id:
                    return state
            for state in self._closed_states.values():
                if state.lifecycle_id == lifecycle_id:
                    return state
            return None

    def update(self, state: ProtectivePositionState) -> None:
        """
        Atomically updates the state. Handles progression from active to CLOSED.
        """
        with self._lock:
            pos_id = state.position_id
            
            # Find the existing state
            existing = self.get(pos_id)
            if not existing:
                raise PositionNotFoundError(
                    f"Cannot update state for position {pos_id}: not found"
                )
                
            # Verify status transition
            if existing.status == PositionLifecycleStatus.CLOSED and state.status != PositionLifecycleStatus.CLOSED:
                raise PositionStateError(
                    f"Cannot transition position {pos_id} out of CLOSED state"
                )
                
            # If transitioning to CLOSED
            if state.status == PositionLifecycleStatus.CLOSED:
                if pos_id in self._active_states:
                    del self._active_states[pos_id]
                self._add_to_closed(state)
            else:
                self._active_states[pos_id] = state

    def close(self, position_id: str, timestamp: str) -> None:
        """
        Closes a position lifecycle atomically.
        """
        with self._lock:
            existing = self.get(position_id)
            if not existing:
                raise PositionNotFoundError(f"Position {position_id} not found to close")
                
            if existing.status == PositionLifecycleStatus.CLOSED:
                return  # Already closed
                
            # Create updated state
            # Note: We cannot modify state since it is frozen = True. We create a copy.
            from dataclasses import replace
            closed_state = replace(
                existing,
                status=PositionLifecycleStatus.CLOSED,
                updated_at=timestamp
            )
            
            self.update(closed_state)

    def invalidate(self, position_id: str) -> None:
        """
        Removes a position state entirely from the store.
        """
        with self._lock:
            if position_id in self._active_states:
                del self._active_states[position_id]
            if position_id in self._closed_states:
                del self._closed_states[position_id]

    def clear(self) -> None:
        """
        Clears all states from the store.
        """
        with self._lock:
            self._active_states.clear()
            self._closed_states.clear()

    def _add_to_closed(self, state: ProtectivePositionState) -> None:
        pos_id = state.position_id
        # OrderedDict maintains insertion order.
        # If it exists, remove it first to place at the end.
        if pos_id in self._closed_states:
            del self._closed_states[pos_id]
        
        self._closed_states[pos_id] = state
        
        # Evict oldest if capacity exceeded
        if len(self._closed_states) > self._max_closed_capacity:
            self._closed_states.popitem(last=False)
            
    def get_all_active(self) -> Dict[str, ProtectivePositionState]:
        with self._lock:
            return dict(self._active_states)
