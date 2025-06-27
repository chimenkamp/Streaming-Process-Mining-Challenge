import abc
from typing import Dict, Any, Optional, List
import time
import pandas as pd


class BaseAlgorithm(abc.ABC):
    """
    Abstract base class for streaming process mining algorithms.

    All algorithms submitted to the challenge must inherit from this class
    and implement the required abstract methods for learning and conformance checking.
    """

    def __init__(self):
        """Initialize the algorithm. Override this method to set up your algorithm."""
        self.start_time = time.time()
        self.processed_events = 0
        self.learning_events = 0
        self.conformance_events = 0
        self.total_processing_time = 0.0
        self.is_learning_phase = True

    @abc.abstractmethod
    def learn(self, event: Dict[str, Any]) -> None:
        """
        Learns from a given event and updates the model.

        This method is called during the warm-up phase (first 10% of the stream)
        where the algorithm can learn patterns without being evaluated.

        Args:
            event (Dict[str, Any]): A dictionary representing an event.
                Common fields include:
                - 'case:concept:name': Case identifier
                - 'concept:name': Activity name
                - 'time:timestamp': Event timestamp
                - 'concept:origin': Origin log (Log A or Log B) for concept drift streams
                - Additional domain-specific attributes

        Example:
            >>> event = {
            ...     'case:concept:name': 'case_001',
            ...     'concept:name': 'Create Application',
            ...     'time:timestamp': '2023-01-01 10:00:00',
            ...     'concept:origin': 'Log A'
            ... }
            >>> algorithm.learn(event)
        """
        pass

    @abc.abstractmethod
    def conformance(self, event: Dict[str, Any]) -> float:
        """
        Checks the conformance of a given event against the learned model.

        This method is called for each event in the full stream and should return
        a conformance value between 0.0 and 1.0, where 1.0 indicates perfect
        conformance and 0.0 indicates complete non-conformance.

        Args:
            event (Dict[str, Any]): A dictionary representing an event.

        Returns:
            float: Conformance value between 0.0 and 1.0

        Example:
            >>> event = {
            ...     'case:concept:name': 'case_001',
            ...     'concept:name': 'Review Application',
            ...     'time:timestamp': '2023-01-01 11:00:00',
            ...     'concept:origin': 'Log A'
            ... }
            >>> conformance = algorithm.conformance(event)
            >>> print(conformance)  # e.g., 0.85
        """
        pass

    def _update_timing(self, event_start_time: float, event_end_time: float):
        """
        Internal method to update timing statistics.

        Args:
            event_start_time (float): Start time of event processing
            event_end_time (float): End time of event processing
        """
        self.processed_events += 1
        self.total_processing_time += (event_end_time - event_start_time)

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics of the algorithm.

        Returns:
            Dict[str, float]: Dictionary with performance metrics
        """
        avg_processing_time = (self.total_processing_time / self.processed_events
                               if self.processed_events > 0 else 0.0)

        return {
            'avg_processing_time': avg_processing_time,
            'total_events_processed': self.processed_events,
            'learning_events': self.learning_events,
            'conformance_events': self.conformance_events,
            'total_runtime': time.time() - self.start_time
        }

    def reset_metrics(self):
        """Reset timing and performance metrics."""
        self.start_time = time.time()
        self.processed_events = 0
        self.learning_events = 0
        self.conformance_events = 0
        self.total_processing_time = 0.0
        self.is_learning_phase = True

    # Optional methods that can be overridden

    def on_learning_phase_start(self, stream_info: Dict[str, Any]):
        """
        Called when the learning phase starts.

        Args:
            stream_info (Dict[str, Any]): Information about the stream
        """
        self.is_learning_phase = True

    def on_learning_phase_end(self, stream_info: Dict[str, Any]):
        """
        Called when the learning phase ends and conformance checking begins.

        Args:
            stream_info (Dict[str, Any]): Information about the learning phase
        """
        self.is_learning_phase = False

    def on_stream_complete(self, stream_info: Dict[str, Any]):
        """
        Called when the entire stream has been processed.

        Args:
            stream_info (Dict[str, Any]): Final stream statistics
        """
        pass


class EventStream:
    """
    Manages event streams for process mining algorithms.

    Handles the separation between learning phase (warm-up) and conformance checking phase.
    Works with both synthetic streams and concept drift streams generated from PNML files.
    """

    def __init__(self, event_log: pd.DataFrame, ground_truth_ratio: float = 0.1):
        """
        Initialize the event stream.

        Args:
            event_log (pd.DataFrame): The complete event log
            ground_truth_ratio (float): Ratio of events to use for learning phase
        """
        self.event_log = event_log.copy()
        self.ground_truth_ratio = ground_truth_ratio

        # Ensure timestamps are datetime objects
        if 'time:timestamp' in self.event_log.columns:
            self.event_log['time:timestamp'] = pd.to_datetime(self.event_log['time:timestamp'])
            self.event_log = self.event_log.sort_values('time:timestamp').reset_index(drop=True)

        # Calculate split point
        self.learning_split = int(len(self.event_log) * self.ground_truth_ratio)

    def get_ground_truth_events(self) -> List[Dict[str, Any]]:
        """Get events for the learning phase (first 10% of stream)."""
        return self.event_log.iloc[:self.learning_split].to_dict('records')

    def get_full_stream(self) -> List[Dict[str, Any]]:
        """Get all events in the stream."""
        return self.event_log.to_dict('records')

    def get_conformance_events(self) -> List[Dict[str, Any]]:
        """Get events for conformance checking (after learning phase)."""
        return self.event_log.iloc[self.learning_split:].to_dict('records')

    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about the stream."""
        return {
            'total_events': len(self.event_log),
            'learning_events': self.learning_split,
            'conformance_events': len(self.event_log) - self.learning_split,
            'learning_ratio': self.ground_truth_ratio,
            'unique_cases': self.event_log[
                'case:concept:name'].nunique() if 'case:concept:name' in self.event_log.columns else 0,
            'unique_activities': self.event_log[
                'concept:name'].nunique() if 'concept:name' in self.event_log.columns else 0,
            'has_concept_drift': 'concept:origin' in self.event_log.columns,
            'origins': self.event_log[
                'concept:origin'].unique().tolist() if 'concept:origin' in self.event_log.columns else [],
            'time_span': {
                'start': self.event_log['time:timestamp'].min() if 'time:timestamp' in self.event_log.columns else None,
                'end': self.event_log['time:timestamp'].max() if 'time:timestamp' in self.event_log.columns else None
            }
        }


import sys
import abc
import time
from collections import defaultdict
from typing import Dict, Any

from algorithm_base import BaseAlgorithm


class BehavioralConformanceAlgorithm(BaseAlgorithm):
    def __init__(self):
        super().__init__()

        # Model builder attributes
        self.__B = []
        self.__P = dict()
        self.__F = dict()
        self.__tracelogs = defaultdict(list)
        self.__mark = []  # used to find P_max for all relation
        self.__mark2 = []

        # Conformance checker attributes
        self.__model_set = False
        self.__trace_last_event = dict()  # recalls last event for trace to find relation
        self.__conformance = dict()  # saves a traces' conformance
        self.__completeness = dict()  # saves a traces' completeness
        self.__confidence = dict()  # saves a traces' confidence
        self.__obs = defaultdict(list)  # saves all distinct relations in a trace that has occurred
        self.__inc = dict()  # saves amount of incorrect relations according to the reference model of a trace
        self.__maxOfMinRelationsAfter = 0

    def learn(self, event: Dict[str, Any]) -> None:
        """Learn from event during training phase using model builder logic."""
        self.learning_events += 1

        case_id = event['case:concept:name']  # locally save caseID
        event_name = event['concept:name']  # locally save event

        if case_id in self.__tracelogs:  # if this caseID has occurred before
            relation = (self.__tracelogs[case_id][-1], event_name)  # find relation based on former event
            traceLength = len(self.__tracelogs[case_id]) - 1  # find the length of the trace so far
            if relation not in self.__B:  # if this relation doesn't exist yet
                self.__B.append(relation)  # add relation
        else:  # if the trace hasn't been seen yet
            self.__tracelogs[case_id] = []

        self.__tracelogs[case_id].append(event_name)  # add the event to the trace log

    def conformance(self, event: Dict[str, Any]) -> float:
        """Check conformance of event against learned model."""
        self.conformance_events += 1

        if not self.__model_set:
            return 0.0  # No model available yet

        case_id = event['case:concept:name']  # easier reference to caseID
        event_name = event['concept:name']  # easier reference to eventName

        if case_id not in self.__trace_last_event:  # if this is first time caseID appears
            self.__trace_last_event[case_id] = event_name  # save current event
            self.__inc[case_id] = 0  # set amount of incorrect relations for that CaseId to 0
            return 1.0  # First event is always conformant

        else:  # if the caseID has been seen before
            new_pattern = (self.__trace_last_event[case_id], event_name)  # locally save the relation

            # Step 1: update internal data structures
            if new_pattern in self.__B:  # if the relation is in the approved relation list
                if new_pattern not in self.__obs[case_id]:  # and that relation has not occurred for that caseID before
                    self.__obs[case_id].append(new_pattern)  # save the relation to that CaseId
            else:  # if the relation is "illegal" according to B
                self.__inc[case_id] += 1  # increment incorrect for that caseID

            # Step 2: compute online conformance values
            self.__conformance[case_id] = len(self.__obs[case_id]) / (
                    len(self.__obs[case_id]) + self.__inc[case_id])  # calculated conformance

            if new_pattern in self.__B:  # if the relation is legal in B
                # if the relation occurrence is within P_min and P_max
                if self.__P[new_pattern][0] <= len(self.__obs[case_id]) and len(self.__obs[case_id]) <= \
                        self.__P[new_pattern][1]:
                    self.__completeness[case_id] = 1  # set completeness for that caseID
                else:  # if not within P_min and P_max
                    self.__completeness[case_id] = min(1, len(self.__obs[case_id]) / (
                            self.__P[new_pattern][0] + 1))  # calculate completeness

                self.__confidence[case_id] = 1 - (self.__F[new_pattern] / self.__maxOfMinRelationsAfter)
            else:
                self.__confidence[case_id] = -1
                self.__completeness[case_id] = -1

            # Step 3: cleanup
            self.__trace_last_event[case_id] = event_name

            return self.__conformance[case_id]

    def on_learning_phase_end(self, stream_info: Dict[str, Any]):
        """Called when learning phase ends - finalize the model."""
        super().on_learning_phase_end(stream_info)
        self._end_xes_to_model()
        self._set_model((self.__B, self.__P, self.__F))

    def _set_model(self, M):
        """Set the model for conformance checking."""
        self.__B = M[0]
        self.__P = M[1]
        self.__F = M[2]
        self.__maxOfMinRelationsAfter = 0
        for relations in self.__F:
            self.__maxOfMinRelationsAfter = max(self.__F[relations], self.__maxOfMinRelationsAfter)
        self.__model_set = True

    def _setF(self, relation):
        """Set F values using breadth-first search from an accepting state."""
        Queue = []  # breath first search from an accepting state
        BFSMark = []
        BFSMark.append(relation)
        Queue.append((relation, 0))
        while len(Queue) > 0:
            ((A, B), depth) = Queue.pop(0)
            self.__F[(A, B)] = min(self.__F[(A, B)], depth)
            for (C, D) in self.__B:
                if D == A and (C, D) not in BFSMark:
                    BFSMark.append((C, D))
                    Queue.append(((C, D), depth + 1))

    def _findP_max(self, relation, depth):
        """Find P_max using brute force to find longest path."""
        if relation not in self.__mark:  # brute force to find longest path
            self.__mark.append(relation)

            (P_min, P_max) = self.__P[relation]
            self.__P[relation] = (P_min, max(depth, P_max))

            (A, B) = relation
            for (C, D) in self.__B:
                if C == B and depth <= 5:
                    self._findP_max((C, D), depth + 1)

            self.__mark.remove(relation)
        else:
            if relation in self.__mark2:
                return
            self.__mark2.append(relation)

            (P_min, P_max) = self.__P[relation]
            self.__P[relation] = (P_min, max(depth, P_max))
            (A, B) = relation
            for (C, D) in self.__B:
                if C == B and depth <= 5:
                    self._findP_max((C, D), depth)

            self.__mark2.remove(relation)

    def _setP(self, relation):
        """Set P values using breadth-first search from beginning state."""
        Queue = []  # breath first state from beginning state
        BFSMark = []
        BFSMark.append(relation)
        Queue.append((relation, 0))
        while len(Queue) > 0:
            ((A, B), depth) = Queue.pop(0)
            (P_min, P_max) = self.__P[(A, B)]
            self.__P[(A, B)] = (min(depth, P_min), max(depth, P_max))
            for (C, D) in self.__B:
                if C == B and (C, D) not in BFSMark:
                    BFSMark.append((C, D))
                    Queue.append(((C, D), depth + 1))

        self._findP_max(relation, 0)

    def _end_xes_to_model(self):
        """Finalize the model after learning phase."""
        for relation in self.__B:
            self.__F[relation] = sys.maxsize
            self.__P[relation] = (sys.maxsize, -1)

        endState = []
        startState = []

        for caseID, trace in self.__tracelogs.items():
            if trace[-1] not in endState:
                endState.append(trace[-1])
            if trace[0] not in startState:
                startState.append(trace[0])

        for (A, B) in self.__B:
            if A in startState:
                self._setP((A, B))

        for (A, B) in self.__B:
            if B in endState:
                self._setF((A, B))

    def get_conformance(self, case_id=None):
        """Get conformance values."""
        if case_id is None:
            return self.__conformance
        return self.__conformance.get(case_id, 0.0)

    def get_completeness(self, case_id=None):
        """Get completeness values."""
        if case_id is None:
            return self.__completeness
        return self.__completeness.get(case_id, 0.0)

    def get_confidence(self, case_id=None):
        """Get confidence values."""
        if case_id is None:
            return self.__confidence
        return self.__confidence.get(case_id, 0.0)

    def get_model(self):
        """Get the learned model."""
        return (self.__B, self.__P, self.__F)