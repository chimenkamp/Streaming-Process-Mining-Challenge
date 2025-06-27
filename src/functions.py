import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import numpy as np
from typing import Sequence, Union, Tuple, List

# Type aliases for clarity (optional but can be helpful)
Numeric = Union[int, float]
NumericSequence = Sequence[Numeric]
Figsize = Tuple[Numeric, Numeric]

def plot_conformance_comparison(
    baseline: NumericSequence,
    conformance: NumericSequence,
    global_errors: Numeric,
    figsize: Figsize = (12, 6),
    linewidth: Numeric = 1
) -> None:
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(baseline, label="Baseline", linestyle='--', linewidth=1)
    ax.plot(conformance, label="Algorithm A", linewidth=linewidth)

    ax.set_xlabel("Event Index")
    ax.set_ylabel("Conformance")
    ax.set_title("Conformance Comparison")

    ax.text(0.5, 0.95, f"Global errors: {global_errors}",
            fontsize=12,
            color='red',
            ha='center',
            va='top',
            transform=ax.transAxes)

    ax.legend()
    plt.tight_layout()
    plt.show()


def tumbling_window_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculate the average of the input list over tumbling windows and return
    a new list where each element is the average of its corresponding window.

    :param data: List of floats to calculate the tumbling window averages from.
    :param window_size: Size of each tumbling window (must be a positive integer).
    :return: A new list of floats where each position contains the average of
             its window from the input list.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    n: int = len(data)
    result: List[float] = [0.0] * n

    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        window = data[start:end]
        window_sum: float = sum(window)
        window_count: int = len(window)
        window_avg: float = window_sum / window_count if window_count > 0 else 0.0

        for i in range(start, end):
            result[i] = window_avg

    return result