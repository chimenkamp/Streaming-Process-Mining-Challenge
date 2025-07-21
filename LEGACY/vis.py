import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
total_time = 100
time = np.linspace(0, total_time, 1000)


def conformance_with_drift(t: np.ndarray, midpoint: float = total_time / 2, lag: float = 1.0,
                           algorithm_num: int = 1) -> tuple[np.ndarray, float]:
    result = np.ones_like(t)
    drift_start = midpoint + lag
    mask = t > drift_start

    if algorithm_num == 1:
        result[mask] = 1 - ((t[mask] - drift_start) / (total_time - drift_start))
        noise = 0.05 * np.sin(1 * t) + 0.03 * np.sin(2 * t + 0.5)
    else:
        result[mask] = 1 - ((t[mask] - drift_start) / (total_time - drift_start)) * 1.3
        noise = 0.08 * np.sin(1.5 * t) + 0.05 * np.sin(2.5 * t + 0.7)

    result = np.clip(result + noise, 0, 1)
    return result, drift_start


y1, drift_start_point1 = conformance_with_drift(time, lag=0.5, algorithm_num=1)
y2, drift_start_point2 = conformance_with_drift(time, lag=0.3, algorithm_num=2)

anomaly_idx = np.where(time >= 40)[0][0]
anomaly_width = 1.5
for i in range(len(time)):
    distance = abs(time[i] - 40)
    if distance < anomaly_width:
        effect = 0.25 * (1 - distance / anomaly_width) ** 2
        y1[i] = max(0.75, y1[i] - effect)
        y2[i] = max(0.72, y2[i] - effect)

baseline = np.ones_like(time)
baseline[(time > total_time / 2) & (time <= 55)] = 0.5
baseline[time > 55] = 0

line1, = ax1.plot(time, y1, 'b-', label='Algorithm 1', linewidth=2)
line2, = ax1.plot(time, y2, 'r-', label='Algorithm 2', linewidth=2)
baseline_line, = ax1.plot(time, baseline, 'g--', label='Baseline', linewidth=2)

ax1.set_xlim(0, total_time)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('Time')
ax1.set_ylabel('Conformance')
ax1.set_title('Stream Conformance with Drift vs Baseline')
ax1.legend()
ax1.grid(True)

ax1.text(total_time / 4, 0.9, "High Conformance", fontsize=10, color='green')
ax1.axhspan(0, 0.2, alpha=0.2, color='red')
ax1.axhspan(0.8, 1, alpha=0.2, color='green')
ax1.axvspan(0, 20, alpha=0.2, color='orange')
ax1.text(10, 0.5, "Warm Up Phase", fontsize=12, color='darkorange',
         ha='center', va='center', rotation=90, fontweight='bold')


def find_time_for_y_value(y_value: float, times: np.ndarray, y_values: np.ndarray,
                          after_time: float = 0) -> float | None:
    valid_indices = np.where(times > after_time)[0]

    for i in range(len(valid_indices) - 1):
        idx = valid_indices[i]
        next_idx = valid_indices[i + 1]

        if (y_values[idx] >= y_value and y_values[next_idx] <= y_value) or \
                (y_values[idx] <= y_value and y_values[next_idx] >= y_value):
            t1, t2 = times[idx], times[next_idx]
            y1, y2 = y_values[idx], y_values[next_idx]

            if y1 == y2:
                return t1

            t = t1 + (y_value - y1) * (t2 - t1) / (y2 - y1)
            return t

    return None


baseline_drop_time = total_time / 2
baseline_drop_to_zero = 55
y_high = 0.7
y_low = 0.3

high_cross_time1 = find_time_for_y_value(y_high, time, y1, after_time=baseline_drop_time)
low_cross_time1 = find_time_for_y_value(y_low, time, y1, after_time=baseline_drop_to_zero)
high_cross_time2 = find_time_for_y_value(y_high, time, y2, after_time=baseline_drop_time)
low_cross_time2 = find_time_for_y_value(y_low, time, y2, after_time=baseline_drop_to_zero)

if high_cross_time1:
    ax1.annotate('',
                 xy=(baseline_drop_time, y_high), xycoords='data',
                 xytext=(high_cross_time1, y_high), textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax1.text((baseline_drop_time + high_cross_time1) / 2, y_high + 0.05,
             f'Lag 1: {high_cross_time1 - baseline_drop_time:.1f}', color='blue', ha='center', fontweight='bold')
    ax1.axvline(x=40, color='m', linestyle='--', alpha=0.7)
    ax1.text(42, 0.9, 'Outlier (Anomaly)', color='m', rotation=90, fontweight='bold')

if high_cross_time2:
    ax1.annotate('',
                 xy=(baseline_drop_time, y_high - 0.05), xycoords='data',
                 xytext=(high_cross_time2, y_high - 0.05), textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text((baseline_drop_time + high_cross_time2) / 2, y_high - 0.1,
             f'Lag 1: {high_cross_time2 - baseline_drop_time:.1f}', color='red', ha='center', fontweight='bold')

if low_cross_time1:
    ax1.annotate('',
                 xy=(baseline_drop_to_zero, y_low), xycoords='data',
                 xytext=(low_cross_time1, y_low), textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax1.text((baseline_drop_to_zero + low_cross_time1) / 2, y_low + 0.05,
             f'Lag 2: {low_cross_time1 - baseline_drop_to_zero:.1f}', color='blue', ha='center', fontweight='bold')

if low_cross_time2:
    ax1.annotate('',
                 xy=(baseline_drop_to_zero, y_low - 0.05), xycoords='data',
                 xytext=(low_cross_time2, y_low - 0.05), textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text((baseline_drop_to_zero + low_cross_time2) / 2, y_low - 0.1,
             f'Lag 2: {low_cross_time2 - baseline_drop_to_zero:.1f}', color='red', ha='center', fontweight='bold')


def compute_computational_time(conformance_values: np.ndarray, baseline_times: list[float],
                               is_algorithm2: bool = False) -> np.ndarray:
    comp_time = np.ones_like(conformance_values) * 10

    if is_algorithm2:
        comp_time *= 1.5

    rate_of_change = np.abs(np.gradient(conformance_values))
    comp_time += rate_of_change * 2000

    warm_up_end_idx = np.argmin(np.abs(time - 20))

    for j in range(0, warm_up_end_idx):
        progress = j / warm_up_end_idx
        warm_up_factor = 120 if is_algorithm2 else 90
        comp_time[j] = warm_up_factor * (1 - (progress * 0.7))

    to_num = 100 if is_algorithm2 else 50

    for i, gr in enumerate(comp_time[0:to_num]):
        comp_time[i] = (120 if is_algorithm2 else 90) - (i*0.1)

    for t in baseline_times:
        idx = np.argmin(np.abs(time - t))
        window = 50
        for j in range(max(0, idx - window), min(len(comp_time), idx + window)):
            distance = abs(j - idx)
            if distance < window:
                spike_factor = 40 if is_algorithm2 else 30
                comp_time[j] += spike_factor * (1 - distance / window)

    anomaly_idx = np.argmin(np.abs(time - 40))
    anomaly_window = 10
    for j in range(max(0, anomaly_idx - anomaly_window), min(len(comp_time), anomaly_idx + anomaly_window)):
        distance = abs(j - anomaly_idx)
        if distance < anomaly_window:
            anomaly_factor = 25 if is_algorithm2 else 20
            comp_time[j] += anomaly_factor * (1 - distance / anomaly_window) ** 2

    return comp_time


comp_time1 = compute_computational_time(y1, [baseline_drop_time, baseline_drop_to_zero], is_algorithm2=False)
comp_time2 = compute_computational_time(y2, [baseline_drop_time, baseline_drop_to_zero], is_algorithm2=True)

ax2.plot(time, comp_time1, 'b-', label='Algorithm 1', linewidth=2)
ax2.plot(time, comp_time2, 'r-', label='Algorithm 2', linewidth=2)

ax2.set_xlim(0, total_time)
ax2.set_ylim(0, max(np.max(comp_time1), np.max(comp_time2)) * 1.1)
ax2.set_ylabel('Computational Time (ms)')
ax2.set_xlabel('Time')
ax2.set_title('Computational Time for Conformance Algorithms')
ax2.legend()
ax2.grid(True)

ax2.axvline(x=baseline_drop_time, color='g', linestyle='--', alpha=0.5)
ax2.axvline(x=baseline_drop_to_zero, color='g', linestyle='--', alpha=0.5)
ax2.axvline(x=40, color='m', linestyle='--', alpha=0.7)

ax2.text(baseline_drop_time + 2, max(comp_time2) * 0.9, 'First Drift', color='g', rotation=90)
ax2.text(baseline_drop_to_zero + 2, max(comp_time2) * 0.9, 'Second Drift', color='g', rotation=90)
ax2.text(42, max(comp_time2) * 0.9, 'Outlier (Anomaly)', color='m', rotation=90)

ax2.axvspan(0, 20, alpha=0.2, color='orange')
ax2.text(10, max(comp_time2) * 0.5, "Warm Up Phase", fontsize=12, color='darkorange',
         ha='center', va='center', rotation=90, fontweight='bold')

plt.tight_layout()
plt.savefig('conformance_drift_with_computation.png', dpi=300, bbox_inches='tight')
plt.show()
