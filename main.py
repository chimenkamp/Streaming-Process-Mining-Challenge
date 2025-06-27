from typing import List

import pandas as pd
import pm4py

from src.functions import plot_conformance_comparison
from src.ui.algorithm_base import EventStream

from tests.behavioral_conformance import BehavioralConformanceAlgorithm

EVENT_LOG_PATH: str = "assets/data/streaming_conformance.xes"

FIRST_MODEL_PATH: str = "assets/data/model_one.pnml"
SECOND_MODEL_PATH: str = "assets/data/model_two.pnml"

# cdlg: ConceptDriftLogGenerator = ConceptDriftLogGenerator(
#     pm4py.convert_to_dataframe(
#         pm4py.play_out(
#             *pm4py.read_pnml(FIRST_MODEL_PATH, auto_guess_final_marking=True)
#         )
#     ),
#     pm4py.convert_to_dataframe(
#         pm4py.play_out(
#             *pm4py.read_pnml(SECOND_MODEL_PATH, auto_guess_final_marking=True)
#         )
#     ),
#     drift_begin_percentage=0.3,
#     drift_end_percentage=0.7,
#     event_level_drift=True
# )

EVENT_LOG: pd.DataFrame = pm4py.read_xes(EVENT_LOG_PATH)
# EVENT_LOG = cdlg.generate_log()

algorithm_a = BehavioralConformanceAlgorithm()

event_stream = EventStream(EVENT_LOG)

ground_truth_events = event_stream.get_ground_truth_events()

full_event_stream = event_stream.get_full_stream()


for e in ground_truth_events:
    algorithm_a.learn(e)

conformance: List[float] = []
for e in full_event_stream:
    con: float = algorithm_a.conformance(e)
    conformance.append(con)


global_errors: int = 0

for i in range(len(ground_truth_events)):
    if conformance[i] != baseline[i]:
        global_errors += 1


baseline: List[float] = baseline
algorithm_a_conformance: List[float] = conformance

plot_conformance_comparison(
    baseline,
    algorithm_a_conformance,
    global_errors,
    figsize=(12, 6),
    linewidth=0.1
)
