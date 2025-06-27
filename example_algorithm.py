"""
Example Algorithm Implementation for Streaming Process Mining Challenge

This file demonstrates how to implement a conformance checking algorithm
that can be uploaded to the challenge platform.

Required libraries:
pandas
numpy

Author: Example Team
Description: A simple frequency-based conformance checking algorithm
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


class AdaptiveConformanceAlgorithm(BaseAlgorithm):
    """
    An adaptive conformance checking algorithm that adjusts to concept drift.
    
    This algorithm maintains a sliding window of recent patterns and
    gradually adapts to changing process behavior.
    """
    
    def __init__(self, window_size: int = 1000, adaptation_rate: float = 0.01):
        """
        Initialize the adaptive algorithm.
        
        Args:
            window_size: Size of the sliding window for adaptation
            adaptation_rate: Rate at which to adapt to new patterns
        """
        super().__init__()
        
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Pattern storage
        self._pattern_buffer = []  # Sliding window of recent patterns
        self._base_patterns = defaultdict(float)  # Base patterns from learning
        self._current_patterns = defaultdict(float)  # Current adapted patterns
        
        # Case tracking
        self._case_contexts = {}  # Context for each case
        
        # Constants
        self.CASE_ID_KEY = 'case:concept:name'
        self.ACTIVITY_KEY = 'concept:name'
    
    def learn(self, event: Dict[str, Any]) -> None:
        """Learn base patterns during learning phase."""
        self.learning_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            return
        
        # Extract pattern
        pattern = self._extract_pattern(case_id, activity)
        if pattern:
            self._base_patterns[pattern] += 1.0
        
        # Update case context
        if case_id not in self._case_contexts:
            self._case_contexts[case_id] = []
        self._case_contexts[case_id].append(activity)
    
    def conformance(self, event: Dict[str, Any]) -> float:
        """Calculate adaptive conformance score."""
        self.conformance_events += 1
        
        case_id = event.get(self.CASE_ID_KEY)
        activity = event.get(self.ACTIVITY_KEY)
        
        if not case_id or not activity:
            return 0.5
        
        # Extract current pattern
        pattern = self._extract_pattern(case_id, activity)
        
        if not pattern:
            return 0.5
        
        # Calculate conformance based on current patterns
        conformance = self._calculate_adaptive_conformance(pattern)
        
        # Update sliding window and adapt
        self._update_patterns(pattern, conformance)
        
        # Update case context
        if case_id not in self._case_contexts:
            self._case_contexts[case_id] = []
        self._case_contexts[case_id].append(activity)
        
        return conformance
    
    def _extract_pattern(self, case_id: str, activity: str) -> Optional[str]:
        """Extract pattern from current event and case context."""
        if case_id not in self._case_contexts or not self._case_contexts[case_id]:
            return f"START->{activity}"
        
        # Use last activity as context
        prev_activity = self._case_contexts[case_id][-1]
        return f"{prev_activity}->{activity}"
    
    def _calculate_adaptive_conformance(self, pattern: str) -> float:
        """Calculate conformance based on adapted patterns."""
        base_score = self._current_patterns.get(pattern, 0.0)
        
        # Normalize and convert to probability
        total_weight = sum(self._current_patterns.values())
        if total_weight > 0:
            probability = base_score / total_weight
        else:
            probability = 0.0
        
        # Scale to [0, 1] range
        return min(1.0, probability * 10)  # Scale factor for better discrimination
    
    def _update_patterns(self, pattern: str, conformance: float):
        """Update patterns using sliding window and adaptation."""
        # Add to buffer
        self._pattern_buffer.append((pattern, conformance))
        
        # Maintain window size
        if len(self._pattern_buffer) > self.window_size:
            self._pattern_buffer.pop(0)
        
        # Update current patterns with adaptation
        self._adapt_patterns()
    
    def _adapt_patterns(self):
        """Adapt patterns based on recent observations."""
        # Start with base patterns
        self._current_patterns = self._base_patterns.copy()
        
        # Adapt based on recent patterns
        recent_patterns = defaultdict(float)
        for pattern, conformance in self._pattern_buffer[-100:]:  # Use last 100 observations
            recent_patterns[pattern] += conformance
        
        # Blend base and recent patterns
        for pattern, recent_weight in recent_patterns.items():
            if pattern in self._current_patterns:
                # Adaptive blending
                old_weight = self._current_patterns[pattern]
                new_weight = (1 - self.adaptation_rate) * old_weight + self.adaptation_rate * recent_weight
                self._current_patterns[pattern] = new_weight
            else:
                # New pattern
                self._current_patterns[pattern] = self.adaptation_rate * recent_weight
    
    def on_learning_phase_end(self, stream_info: Dict[str, Any]):
        """Initialize current patterns from base patterns."""
        super().on_learning_phase_end(stream_info)
        
        # Normalize base patterns
        total = sum(self._base_patterns.values())
        if total > 0:
            for pattern in self._base_patterns:
                self._base_patterns[pattern] /= total
        
        # Initialize current patterns
        self._current_patterns = self._base_patterns.copy()


# Example of how to create a simple wrapper class if needed
class SubmissionAlgorithm(FrequencyBasedConformanceAlgorithm):
    """
    Main algorithm class for submission.
    
    This is a simple wrapper around FrequencyBasedConformanceAlgorithm
    that can be easily identified by the platform.
    """
    
    def __init__(self):
        super().__init__()
        # Override default parameters if needed
        self.min_frequency_threshold = 3
        self.smoothing_factor = 0.05
    
    def get_submission_info(self) -> Dict[str, str]:
        """Get information for submission."""
        return {
            'team_name': 'Example Team',
            'algorithm_name': 'Frequency-Based Conformance Checker',
            'version': '1.0',
            'description': 'A frequency-based algorithm that learns transition patterns and assesses conformance based on learned probabilities.'
        }


# The platform will automatically detect and load one of these algorithm classes
# Make sure at least one class inherits from BaseAlgorithm and implements the required methods