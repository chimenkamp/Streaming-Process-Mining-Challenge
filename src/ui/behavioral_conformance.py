import sys
from collections import defaultdict
from typing import Dict, Any, Set, Tuple, List, Optional

from algorithm_base import BaseAlgorithm


class BehavioralConformanceAlgorithm(BaseAlgorithm):

    def __init__(self) -> None:
        """
        Initializes the BehavioralConformanceAlgorithm.
        - _B: Stores the learned behavioral model (set of allowed transitions).
        - _learn_last_event: Tracks the last event per case during the learning phase.
        - _conformance_last_event: Tracks the last event per case during conformance checking.
        - _obs: Tracks observed conforming transitions per case during conformance checking.
        - _inc: Tracks the count of non-conforming transitions per case during conformance checking.
        - _conformance_values: Stores the current conformance value per case.
        """
        super().__init__()
        self._B: Set[Tuple[str, str]] = set()
        self._learn_last_event: Dict[str, str] = {}

        self._conformance_last_event: Dict[str, str] = {}
        self._obs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._inc: Dict[str, int] = defaultdict(int)
        self._conformance_values: Dict[str, float] = defaultdict(lambda: 1.0)

        self.CASE_ID_KEY: str = 'case:concept:name'
        self.ACTIVITY_KEY: str = 'concept:name'


    def _get_case_activity(self, event: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Safely extracts case ID and activity name from an event dictionary."""
        case_id: Optional[str] = event.get(self.CASE_ID_KEY)
        activity: Optional[str] = event.get(self.ACTIVITY_KEY)
        return case_id if isinstance(case_id, str) else None, \
               activity if isinstance(activity, str) else None

    def learn(self, event: Dict[str, Any]) -> None:
        """
        Learns allowed transitions from the event stream.

        Updates the internal model (_B) by observing pairs of directly
        following activities within each case.

        :param event: A dictionary representing an event, expected to have
                      keys for case ID (e.g., 'case:concept:name') and
                      activity name (e.g., 'concept:name').
        """
        case_id: Optional[str]
        current_activity: Optional[str]
        case_id, current_activity = self._get_case_activity(event)

        if case_id is None or current_activity is None:
            print(f"Warning: Event missing case ID or activity during learning: {event}", file=sys.stderr)
            return

        if case_id in self._learn_last_event:
            last_activity: str = self._learn_last_event[case_id]
            relation: Tuple[str, str] = (last_activity, current_activity)
            if relation not in self._B:
                self._B.add(relation)

        self._learn_last_event[case_id] = current_activity
        self.learning_events += 1

    def conformance(self, event: Dict[str, Any]) -> float:
        """
        Calculates the behavioral conformance of the given event.

        Checks if the transition leading to this event is allowed by the
        learned model (_B). Updates and returns the running conformance
        score for the event's case.

        :param event: A dictionary representing an event.
        :return: The calculated conformance value (float between 0.0 and 1.0)
                 for the case associated with the event. Returns 1.0 if it's
                 the first event of a case or if the event lacks necessary info.
        """
        case_id: Optional[str]
        current_activity: Optional[str]
        case_id, current_activity = self._get_case_activity(event)

        if case_id is None or current_activity is None:
            return self._conformance_values.get(case_id, 1.0) if case_id is not None else 1.0

        if case_id not in self._conformance_last_event:
            self._conformance_last_event[case_id] = current_activity
            self._obs[case_id] = []
            self._inc[case_id] = 0
            self._conformance_values[case_id] = 1.0
            self.conformance_events += 1
            return 1.0
        else:
            last_activity: str = self._conformance_last_event[case_id]
            relation: Tuple[str, str] = (last_activity, current_activity)

            if relation in self._B:
                self._obs[case_id].append(relation)
            else:
                self._inc[case_id] += 1

            num_unique_observed_conforming: int = len(set(self._obs[case_id]))
            num_incorrect: int = self._inc[case_id]
            total_relations_considered = num_unique_observed_conforming + num_incorrect

            current_conformance: float
            if total_relations_considered == 0:
                 current_conformance = 1.0 if num_unique_observed_conforming > 0 else 1.0
            else:
                current_conformance = float(num_unique_observed_conforming) / total_relations_considered

            self._conformance_last_event[case_id] = current_activity
            self._conformance_values[case_id] = current_conformance
            self.conformance_events += 1

            return current_conformance