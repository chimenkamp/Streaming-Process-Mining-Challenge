import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any


class ConceptDriftLogGenerator:
    """
    Generates an event log exhibiting concept drift by combining two existing event logs.

    The drift occurs between a specified start and end percentage of the combined
    event sequence, transitioning probabilistically from the characteristics of
    the first log (Log A) to the characteristics of the second log (Log B).
    """

    def __init__(
        self,
        log_a: pd.DataFrame,
        log_b: pd.DataFrame,
        drift_begin_percentage: float,
        drift_end_percentage: float,
    ) -> None:
        """
        Initialize the ConceptDriftLogGenerator.

        :param log_a: DataFrame for the first event log (before drift).
        :param log_b: DataFrame for the second event log (after drift).
        :param drift_begin_percentage: Float between 0.0 and 1.0 indicating where drift starts.
        :param drift_end_percentage: Float between 0.0 and 1.0 indicating where drift ends.
        :return: None.
        """
        if not (0.0 <= drift_begin_percentage <= 1.0):
            raise ValueError("drift_begin_percentage must be between 0.0 and 1.0")
        if not (0.0 <= drift_end_percentage <= 1.0):
            raise ValueError("drift_end_percentage must be between 0.0 and 1.0")
        if drift_end_percentage < drift_begin_percentage:
            raise ValueError("drift_end_percentage cannot be less than drift_begin_percentage")

        self.required_columns = {"case:concept:name", "concept:name", "time:timestamp"}
        self._validate_log(log_a, "Log A")
        self._validate_log(log_b, "Log B")

        self.log_a: pd.DataFrame = log_a.copy()
        self.log_b: pd.DataFrame = log_b.copy()

        self.log_a["time:timestamp"] = pd.to_datetime(self.log_a["time:timestamp"])
        self.log_b["time:timestamp"] = pd.to_datetime(self.log_b["time:timestamp"])

        self.log_a["concept:origin"] = "Log A"
        self.log_b["concept:origin"] = "Log B"

        self.drift_begin: float = drift_begin_percentage
        self.drift_end: float = drift_end_percentage

    def _validate_log(self, log: pd.DataFrame, name: str) -> None:
        """
        Check if the log DataFrame has the required columns.

        :param log: DataFrame to validate.
        :param name: Name identifier for error messages.
        :return: None.
        """
        missing = self.required_columns - set(log.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    def generate_log(self) -> pd.DataFrame:
        """
        Generate the event log with concept drift (event-level).

        :return: DataFrame representing the combined log with drift applied.
        """
        return self._generate_event_level_drift()

    def _generate_event_level_drift(self) -> pd.DataFrame:
        """
        Generate log with event-level drift by interleaving events from log_a and log_b
        based on their original order, applying a gradually increasing probability
        of selecting from Log B between the specified drift begin and end percentages.

        :return: DataFrame with event-level drift applied.
        """
        if self.log_a.empty and self.log_b.empty:
            columns = list(self.required_columns) + ["concept:origin"]
            return pd.DataFrame(columns=columns)

        # Use the original ordering of each log (no time-based sorting).
        events_a: pd.DataFrame = self.log_a.reset_index(drop=True)
        events_b: pd.DataFrame = self.log_b.reset_index(drop=True)

        total_events: int = len(events_a) + len(events_b)
        if total_events == 0:
            columns = list(self.required_columns) + ["concept:origin"]
            return pd.DataFrame(columns=columns)

        pointer_a: int = 0
        pointer_b: int = 0
        selected_events: List[Dict[str, Any]] = []

        for idx in range(total_events):
            progress: float = idx / (total_events - 1) if total_events > 1 else 0.0
            if progress < self.drift_begin:
                prob_b: float = 0.0
            elif progress > self.drift_end:
                prob_b: float = 1.0
            else:
                if self.drift_begin == self.drift_end:
                    prob_b = 0.0
                else:
                    prob_b = (progress - self.drift_begin) / (self.drift_end - self.drift_begin)

            choose_b: bool = random.random() < prob_b

            if choose_b and pointer_b < len(events_b):
                row = events_b.iloc[pointer_b].to_dict()
                pointer_b += 1
            elif (not choose_b) and pointer_a < len(events_a):
                row = events_a.iloc[pointer_a].to_dict()
                pointer_a += 1
            elif pointer_a < len(events_a):
                row = events_a.iloc[pointer_a].to_dict()
                pointer_a += 1
            else:
                row = events_b.iloc[pointer_b].to_dict()
                pointer_b += 1

            selected_events.append(row)

        final_log: pd.DataFrame = pd.DataFrame(selected_events)
        return final_log
