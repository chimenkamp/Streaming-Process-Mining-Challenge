from typing import Dict, Any, List

import pandas as pd
import pm4py
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog, Trace

from pm4py.objects.log.obj import Trace, EventLog, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.conversion.log import converter as log_converter

def check_conformance(trace: Trace | List[Event], net: tuple[PetriNet, Marking, Marking]) -> Dict[str, Any]:
    """
    Checks the token-based replay conformance of a given trace against a Petri net.

    :param trace: A PM4Py Trace object containing a sequence of Event objects with 'concept:name' set.
    :param net_path: Path to the Petri net in PNML format.
    :return: A dictionary containing the token-based replay result (e.g., fitness).
    """
    log: EventLog = EventLog()
    log.append(trace)
    log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    result = token_replay.apply(log, net[0], net[1], net[2])
    return result[0]


LOG_WITH_DRIFT_PATH: str = "../assets/data/streaming_conformance.xes"
FIRST_MODEL_PATH: str = "../assets/data/model_one.pnml"

MODEL_ONE: tuple[PetriNet, Marking, Marking] = pm4py.read_pnml(FIRST_MODEL_PATH)

MODEL_ONE = (MODEL_ONE[0], MODEL_ONE[1], pm4py.generate_marking(MODEL_ONE[0], "cbbb31ae-3d9d-41e0-bca8-374beba17e99"))

LOG_WITH_DRIFT: EventLog = pm4py.read_xes(LOG_WITH_DRIFT_PATH, return_legacy_log_object=True)

fitness_values_trace = []
for trace in LOG_WITH_DRIFT:
    res = check_conformance(trace, MODEL_ONE)
    fitness_values_trace.append(res["trace_fitness"])


WINDOW_SIZE = 5
fitness_values_window = []
for trace in LOG_WITH_DRIFT:
    window = []
    for event in trace:
        window.append(event)

        if len(window) == WINDOW_SIZE:
            trace = Trace()
            for event in window:
                trace.append(event)

            res = check_conformance(trace, MODEL_ONE)
            fitness_values_window.append(res["trace_fitness"])
            window.clear()

# plot the fitness values
import matplotlib.pyplot as plt

plt.plot(fitness_values_trace)
plt.plot(fitness_values_window)
plt.xlabel('Trace Index')
plt.ylabel('Fitness Value')
plt.title('Token-based Replay Fitness Values')
plt.show()
