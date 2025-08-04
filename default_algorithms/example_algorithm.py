"""
Example Algorithm Implementation for Streaming Process Mining Challenge

This file demonstrates how to implement a conformance checking algorithm
that can be uploaded to the challenge platform.

Required libraries:
pandas
numpy

Author: AVOCADO Platform
Description: Default example algorithms for all users
"""

import sys
from collections import defaultdict, Counter
from typing import Dict, Any, Set, Tuple, Optional
import numpy as np

# Note: BaseAlgorithm will be imported automatically by the platform
from algorithm_base import BaseAlgorithm


class FrequencyBasedConformanceAlgorithm(BaseAlgorithm):
    """
    A frequency-based conformance checking algorithm.
    
    This algorithm learns the frequency of activity transitions during the learning phase
    and uses this information to assess conformance during the checking phase.
    Transitions that occur more frequently during learning are considered more conformant.
    """
    
    def __init__(self):
        """Initialize the algorithm."""
        super().__init__()
        
        # Learning phase data structures
        self._transitions = defaultdict(int)  # Count of each transition
        self._activity_counts = defaultdict(int)  # Count of each activity
        self._case_traces = defaultdict(list)  # Store traces during learning
        
        # Conformance checking data structures
        self._transition_probabilities = {}  # Learned transition probabilities
        self._activity_probabilities = {}  # Learned activity probabilities
        self._case_last_activity = {}  # Track last activity per case
        self._case_conformance_scores = defaultdict(list)  # Store conformance scores per case
        
        # Configuration
        self.min_frequency_threshold = 2  # Minimum frequency to consider a transition valid
        self.smoothing_factor = 0.1  # Laplace smoothing factor
        
        # Constants for event field names
        self.CASE_ID_KEY = 'case:concept:name'
        self.ACTIVITY_KEY = 'concept:name'
    
    def learn(self, event: Dict[str, Any]) -> None:
        """
        Learn patterns from the event during the learning phase.
        
        Args:
            event: Dictionary containing event data with keys:
                  - 'case:concept:name': Case identifier
                  - 'concept:name': Activity name
                  - Additional attributes may be present
        """
        self.learning_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            # Skip events with missing essential information
            return
        
        # Track activity frequency
        self._activity_counts[activity] += 1
        
        # Track transitions
        if case_id in self._case_traces and self._case_traces[case_id]:
            # There's a previous activity in this case
            prev_activity = self._case_traces[case_id][-1]
            transition = (prev_activity, activity)
            self._transitions[transition] += 1
        
        # Add activity to case trace
        self._case_traces[case_id].append(activity)
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """
        Calculate conformance score for the given event.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            Float between 0.0 and 1.0 representing conformance score
        """
        self.conformance_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            return 0.5  # Neutral score for missing data
        
        # Calculate conformance score
        conformance_score = self._calculate_conformance_score(case_id, activity)
        
        # Update tracking for next event
        self._case_last_activity[case_id] = activity
        self._case_conformance_scores[case_id].append(conformance_score)
        
        return conformance_score
    
    def _calculate_conformance_score(self, case_id: str, activity: str) -> float:
        """
        Calculate conformance score based on learned patterns.
        
        Args:
            case_id: Case identifier
            activity: Current activity name
            
        Returns:
            Conformance score between 0.0 and 1.0
        """
        # For the first activity in a case, use activity frequency
        if case_id not in self._case_last_activity:
            return self._get_activity_conformance(activity)
        
        # For subsequent activities, use transition probability
        prev_activity = self._case_last_activity[case_id]
        transition = (prev_activity, activity)
        
        return self._get_transition_conformance(transition, activity)
    
    def _get_activity_conformance(self, activity: str) -> float:
        """Get conformance score based on activity frequency."""
        if activity in self._activity_probabilities:
            # Use learned probability
            return self._activity_probabilities[activity]
        else:
            # Unknown activity - low conformance
            return 0.1
    
    def _get_transition_conformance(self, transition: Tuple[str, str], activity: str) -> float:
        """Get conformance score based on transition probability."""
        if transition in self._transition_probabilities:
            # Use learned transition probability
            return self._transition_probabilities[transition]
        else:
            # Unknown transition - use activity frequency as fallback
            return self._get_activity_conformance(activity) * 0.5
    
    def on_learning_phase_end(self, stream_info: Dict[str, Any]):
        """
        Called when learning phase ends. Calculate probabilities from learned data.
        
        Args:
            stream_info: Information about the learning phase
        """
        super().on_learning_phase_end(stream_info)
        self._calculate_probabilities()
    
    def _calculate_probabilities(self):
        """Calculate transition and activity probabilities from learned data."""
        # Calculate activity probabilities
        total_activities = sum(self._activity_counts.values())
        if total_activities > 0:
            for activity, count in self._activity_counts.items():
                # Apply Laplace smoothing
                prob = (count + self.smoothing_factor) / (total_activities + self.smoothing_factor * len(self._activity_counts))
                self._activity_probabilities[activity] = min(1.0, prob * 2)  # Scale up for better discrimination
        
        # Calculate transition probabilities
        total_transitions = sum(self._transitions.values())
        if total_transitions > 0:
            # Group transitions by source activity
            source_totals = defaultdict(int)
            for (source, target), count in self._transitions.items():
                source_totals[source] += count
            
            # Calculate conditional probabilities
            for (source, target), count in self._transitions.items():
                if count >= self.min_frequency_threshold:
                    # P(target|source)
                    prob = count / source_totals[source] if source_totals[source] > 0 else 0
                    self._transition_probabilities[(source, target)] = min(1.0, prob)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the algorithm and its current state.
        
        Returns:
            Dictionary with algorithm information
        """
        return {
            'name': 'FrequencyBasedConformanceAlgorithm',
            'description': 'Frequency-based conformance checking using transition probabilities',
            'learned_activities': len(self._activity_counts),
            'learned_transitions': len(self._transitions),
            'valid_transitions': len(self._transition_probabilities),
            'min_frequency_threshold': self.min_frequency_threshold,
            'smoothing_factor': self.smoothing_factor
        }
    
    def reset(self):
        """Reset the algorithm to initial state."""
        super().reset_metrics()
        
        # Clear learning data
        self._transitions.clear()
        self._activity_counts.clear()
        self._case_traces.clear()
        
        # Clear conformance data
        self._transition_probabilities.clear()
        self._activity_probabilities.clear()
        self._case_last_activity.clear()
        self._case_conformance_scores.clear()


class SimpleBaselineAlgorithm(BaseAlgorithm):
    """
    A simple baseline algorithm that returns random conformance scores.
    
    This serves as a basic example and baseline for comparison.
    """
    
    def __init__(self):
        """Initialize the baseline algorithm."""
        super().__init__()
        self._random_seed = 42
        self._activity_seen = set()
        
    def learn(self, event: Dict[str, Any]) -> None:
        """Learn from events by simply tracking seen activities."""
        self.learning_events += 1
        
        activity = event.get('concept:name')
        if activity:
            self._activity_seen.add(activity)
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """Return simple conformance based on whether activity was seen in learning."""
        self.conformance_events += 1
        
        activity = event.get('concept:name')
        if not activity:
            return 0.5
        
        # Simple rule: if activity was seen during learning, return high conformance
        if activity in self._activity_seen:
            return 0.8
        else:
            return 0.2
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information."""
        return {
            'name': 'SimpleBaselineAlgorithm',
            'description': 'Simple baseline algorithm for demonstration',
            'activities_learned': len(self._activity_seen)
        }


# The platform will automatically detect and load one of these algorithm classes
# Make sure at least one class inherits from BaseAlgorithm and implements the required methods
