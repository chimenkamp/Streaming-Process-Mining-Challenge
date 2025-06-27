import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

from pm4py.objects.log.obj import EventLog

from test_streams import ConceptDriftLogGenerator


class StreamManager:
    """
    Manages stream configurations and generates concept drift streams from PNML sources
    or synthetic logs as fallback.
    """

    def __init__(self):
        """Initialize the StreamManager with configuration file."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(
            project_root,
            "assets",
            "config",
            "stream_configuration.json"
        )
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load stream configuration from JSON file."""
        try:
            # Try to load from the specified path
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    print(f"Loaded configuration from {self.config_path}")
                    return config
            else:
                print(f"Config file not found at {self.config_path}")

        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
        except Exception as e:
            print(f"Error loading config file: {e}")

        # Fallback to default configuration
        print("Using default configuration")
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when config file is not available."""
        return {
            "streams": [
                {
                    "id": "simple_sequential",
                    "name": "Simple Concept Drift",
                    "description": "A basic stream containing a simple concept drift. The first 60% of the stream follows one model, and the last 40% follows another.",
                    "type": "pnml_drift",
                    "is_testing": True,
                    "pnml_log_a": "data/model_one.pnml",
                    "pnml_log_b": "data/model_two.pnml",
                    "drift_config": {
                        "drift_type": "sudden",
                        "drift_begin_percentage": 0.6,
                        "drift_end_percentage": 0.6,
                        "event_level_drift": False,
                        "num_cases_a": 300,
                        "num_cases_b": 200
                    }
                }
            ]
        }

    def get_testing_streams(self) -> List[Dict[str, str]]:
        """Get list of streams available for testing (is_testing=true)."""
        testing_streams = []
        for stream in self.config.get("streams", []):
            if stream.get("is_testing", False):
                testing_streams.append({
                    'label': stream['name'],
                    'value': stream['id']
                })
        return testing_streams

    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific stream."""
        for stream in self.config.get("streams", []):
            if stream['id'] == stream_id:
                return stream
        return None

    def generate_concept_drift_stream(self, stream_id: str, num_cases: int = 500) -> Tuple[pd.DataFrame, List[float]]:
        """
        Generate a concept drift stream and its baseline conformance values.

        Returns:
            Tuple of (event_log_dataframe, baseline_conformance_list)
        """
        stream_info = self.get_stream_info(stream_id)
        if not stream_info:
            raise ValueError(f"Stream {stream_id} not found")

        return self._generate_pnml_drift_stream(stream_info, num_cases)

    def _generate_pnml_drift_stream(self, stream_info: Dict[str, Any], num_cases: int) -> Tuple[
        pd.DataFrame, List[float]]:
        """Generate concept drift stream from PNML files."""
        drift_config = stream_info['drift_config']

        # Generate or load the two event logs
        log_a, log_b = self._load_or_generate_pnml_logs(
            stream_info['pnml_log_a'],
            stream_info['pnml_log_b'],
            drift_config.get('num_cases_a', num_cases // 2),
            drift_config.get('num_cases_b', num_cases // 2)
        )

        # Create concept drift log
        drift_generator = ConceptDriftLogGenerator(
            log_a=log_a,
            log_b=log_b,
            drift_begin_percentage=drift_config['drift_begin_percentage'],
            drift_end_percentage=drift_config['drift_end_percentage'],
        )

        drifted_log = drift_generator.generate_log()

        # Generate baseline considering drift type
        baseline = self._generate_baseline_with_drift_type(drifted_log, drift_config)

        return drifted_log, baseline

    def _load_or_generate_pnml_logs(self, pnml_a_path: str, pnml_b_path: str,
                                    num_cases_a: int, num_cases_b: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load PNML files or generate synthetic logs if files don't exist."""
        try:
            # Try to import pm4py for PNML support
            import pm4py

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            pnml_a_path = os.path.join(
                project_root,
                "assets",
                "data",
                pnml_a_path
            )
            pnml_b_path = os.path.join(
                project_root,
                "assets",
                "data",
                pnml_b_path
            )

            # Check if files exist
            if pnml_a_path and pnml_b_path:
                print(f"Loading PNML files: {pnml_a_path}, {pnml_b_path}")

                net_a, initial_marking_a, final_marking_a = pm4py.read_pnml(pnml_a_path, auto_guess_final_marking=True)
                net_b, initial_marking_b, final_marking_b = pm4py.read_pnml(pnml_b_path, auto_guess_final_marking=True)

                # Generate event logs from PNML models
                log_a_pm4py: EventLog = pm4py.play_out(net_a, initial_marking_a, final_marking_a,
                                                       parameters={"noTraces": num_cases_a})
                log_b_pm4py: EventLog = pm4py.play_out(net_b, initial_marking_b, final_marking_b,
                                                       parameters={"noTraces": num_cases_b})

                # Convert to our expected format
                log_a = pm4py.convert_to_dataframe(log_a_pm4py)
                log_b = pm4py.convert_to_dataframe(log_b_pm4py)

                # Ensure required columns exist and are properly named
                log_a = self._standardize_log_format(log_a, "Log A")
                log_b = self._standardize_log_format(log_b, "Log B")

                print(f"Successfully generated logs: {len(log_a)} events from Log A, {len(log_b)} events from Log B")
                return log_a, log_b
            else:
                print(f"PNML files not found: {pnml_a_path}, {pnml_b_path}")

        except ImportError:
            print("PM4Py not available. Install with: pip install pm4py")
        except Exception as e:
            print(f"Error loading PNML files: {e}")

    def _standardize_log_format(self, log: pd.DataFrame, origin: str) -> pd.DataFrame:
        """Standardize log format to match expected column names."""
        # Map common PM4Py column names to our expected format
        column_mapping = {
            'case_id': 'case:concept:name',
            'case:concept:name': 'case:concept:name',
            'activity': 'concept:name',
            'concept:name': 'concept:name',
            'timestamp': 'time:timestamp',
            'time:timestamp': 'time:timestamp',
            'start_timestamp': 'time:timestamp'
        }

        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in log.columns and new_name not in log.columns:
                log = log.rename(columns={old_name: new_name})

        # Ensure required columns exist
        required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
        for col in required_columns:
            if col not in log.columns:
                if col == 'case:concept:name':
                    # Generate case IDs if missing
                    unique_traces = log.groupby(log.index // 10).ngroup()  # Assume ~10 events per trace
                    log[col] = [f'case_{origin}_{i:03d}' for i in unique_traces]
                elif col == 'concept:name':
                    log[col] = f'Activity_{origin}'
                elif col == 'time:timestamp':
                    log[col] = pd.date_range('2023-01-01', periods=len(log), freq='1h')

        # Add origin column
        log['concept:origin'] = origin

        # Ensure timestamp is datetime
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

        # Ensure case IDs are unique by adding prefix if needed
        if 'case:concept:name' in log.columns:
            case_ids = log['case:concept:name'].astype(str)
            if not case_ids.str.contains(origin).any():
                log['case:concept:name'] = case_ids + f'_{origin}'

        return log

    def _generate_baseline_with_drift_type(self, drifted_log: pd.DataFrame, drift_config: Dict[str, Any]) -> List[
        float]:
        """
        Generate baseline conformance values considering the drift type.

        For sudden drift: Binary baseline (1.0 for Log A, 0.0 for Log B)
        For gradual drift: Gradual transition during drift period
        """
        baseline = []
        total_events = len(drifted_log)
        drift_type = drift_config.get('drift_type', 'sudden')
        drift_begin = drift_config['drift_begin_percentage']
        drift_end = drift_config['drift_end_percentage']

        for idx, (_, event) in enumerate(drifted_log.iterrows()):
            progress = idx / (total_events - 1) if total_events > 1 else 0.0
            origin = event.get('concept:origin', 'Log A')

            if drift_type == 'gradual' and drift_begin != drift_end:
                # Gradual drift baseline
                if progress < drift_begin:
                    # Before drift: always conforming
                    baseline_value = 1.0
                elif progress > drift_end:
                    # After drift: always non-conforming
                    baseline_value = 0.0
                else:
                    # During drift: linear transition from 1.0 to 0.0
                    drift_progress = (progress - drift_begin) / (drift_end - drift_begin)
                    baseline_value = 1.0 - drift_progress
            else:
                # Sudden drift or when begin == end: binary baseline
                if origin == 'Log A':
                    baseline_value = 1.0
                else:  # Log B
                    baseline_value = 0.0

            baseline.append(baseline_value)

        return baseline

    def _generate_baseline_from_origin(self, drifted_log: pd.DataFrame) -> List[float]:
        """
        Generate baseline conformance values based on event origin.
        Log A events = 1.0 (conforming)
        Log B events = 0.0 (non-conforming)

        NOTE: This method is kept for backward compatibility but should use
        _generate_baseline_with_drift_type for proper gradual drift handling.
        """
        baseline = []
        for _, event in drifted_log.iterrows():
            origin = event.get('concept:origin', 'Log A')
            if origin == 'Log A':
                baseline.append(1.0)
            else:  # Log B
                baseline.append(0.0)

        return baseline

    def get_stream_description(self, stream_id: str) -> str:
        """Get detailed description of a stream."""
        stream_info = self.get_stream_info(stream_id)
        if not stream_info:
            return "Stream not found"

        description = stream_info['description']

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        pnml_a_path = os.path.join(
            project_root,
            "assets",
            "data",
            stream_info['pnml_log_a']
        )
        pnml_b_path = os.path.join(
            project_root,
            "assets",
            "data",
            stream_info['pnml_log_a']
        )

        if stream_info['type'] == 'pnml_drift':
            drift_config = stream_info['drift_config']
            description += f"\n\nDrift Configuration:"
            description += f"\n• Type: {drift_config['drift_type'].title()}"
            description += f"\n• Begin: {drift_config['drift_begin_percentage'] * 100:.1f}%"
            description += f"\n• End: {drift_config['drift_end_percentage'] * 100:.1f}%"
            description += f"\n• Level: {'Event' if drift_config.get('event_level_drift', False) else 'Case'}"

            # Add PNML file information
            if pnml_a_path and pnml_b_path:
                description += f"\n\nPNML Files:"
                description += f"\n• Log A: {pnml_a_path}"
                description += f"\n• Log B: {pnml_b_path}"

                # Check if files exist
                if os.path.exists(pnml_a_path) and os.path.exists(pnml_b_path):
                    description += f"\n• Status: ✓ Files found"
                else:
                    description += f"\n• Status: ⚠ Files missing - using synthetic fallback"

            if drift_config['drift_type'] == 'sudden':
                description += f"\n\nBaseline: Events before drift = 1.0, after drift = 0.0"
            elif drift_config['drift_type'] == 'gradual' and drift_config['drift_begin_percentage'] != drift_config[
                'drift_end_percentage']:
                description += f"\n\nBaseline: Gradual transition from 1.0 to 0.0 during drift period"
            else:
                description += f"\n\nBaseline: Binary transition based on event origin"
        else:
            description += f"\n\nBaseline: All events = 1.0 (no concept drift)"

        return description

    def get_drift_visualization_data(self, stream_id: str, total_events: int) -> Dict[str, Any]:
        """Get data for visualizing concept drift regions."""
        stream_info = self.get_stream_info(stream_id)
        if not stream_info or stream_info['type'] != 'pnml_drift':
            return {}

        drift_config = stream_info['drift_config']

        drift_start = int(drift_config['drift_begin_percentage'] * total_events)
        drift_end = int(drift_config['drift_end_percentage'] * total_events)

        return {
            'has_drift': True,
            'drift_type': drift_config['drift_type'],
            'drift_start': drift_start,
            'drift_end': drift_end,
            'event_level': drift_config.get('event_level_drift', False)
        }