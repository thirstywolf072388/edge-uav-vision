import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/latency/yolo_cpu_timing.csv")

# --- Separate cold start (frame 1) from steady-state performance ---
cold_start_ms = df.iloc[0]["total_ms"]
steady_state = df.iloc[1:]  # everything after the first frame

mean_latency = steady_state["total_ms"].mean()
median_latency = steady_state["total_ms"].median()
p95_latency = steady_state["total_ms"].quantile(0.95)
p99_latency = steady_state["total_ms"].quantile(0.99)
min_latency = steady_state["total_ms"].min()
max_latency = steady_state["total_ms"].max()
mean_fps = steady_state["fps"].mean()

print("--- Cold Start ---")
print(f"First-frame latency: {cold_start_ms:.2f} ms  (model load + graph init, one-time cost)")

print("\n--- Steady-State Latency Summary (YOLOv8n, CPU, WSL2, frames 2+) ---")
print(f"Frames analyzed:   {len(steady_state)}")
print(f"Mean latency:      {mean_latency:.2f} ms")
print(f"Median latency:    {median_latency:.2f} ms")
print(f"p95 latency:       {p95_latency:.2f} ms")
print(f"p99 latency:       {p99_latency:.2f} ms")
print(f"Min / Max latency: {min_latency:.2f} ms / {max_latency:.2f} ms")
print(f"Mean FPS:          {mean_fps:.2f}")

with open("evaluation/latency/summary.txt", "w") as f:
    f.write("YOLOv8n on CPU (WSL2)\n\n")
    f.write(f"Cold start (first frame): {cold_start_ms:.2f} ms\n\n")
    f.write("Steady-state (frames 2+):\n")
    f.write(f"Frames analyzed: {len(steady_state)}\n")
    f.write(f"Mean latency: {mean_latency:.2f} ms\n")
    f.write(f"Median latency: {median_latency:.2f} ms\n")
    f.write(f"p95 latency: {p95_latency:.2f} ms\n")
    f.write(f"p99 latency: {p99_latency:.2f} ms\n")
    f.write(f"Min / Max latency: {min_latency:.2f} / {max_latency:.2f} ms\n")
    f.write(f"Mean FPS: {mean_fps:.2f}\n")

# --- Plot 1: steady-state latency over time ---
plt.figure(figsize=(10, 5))
plt.plot(steady_state["frame"], steady_state["total_ms"], linewidth=0.8)
plt.axhline(mean_latency, color="green", linestyle="--", label=f"Mean ({mean_latency:.1f}ms)")
plt.axhline(p95_latency, color="orange", linestyle="--", label=f"p95 ({p95_latency:.1f}ms)")
plt.axhline(p99_latency, color="red", linestyle="--", label=f"p99 ({p99_latency:.1f}ms)")
plt.xlabel("Frame number")
plt.ylabel("Total latency (ms)")
plt.title("Per-Frame Inference Latency (Steady State) — YOLOv8n on CPU")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/latency/latency_over_time.png", dpi=150)
print("\nSaved plot: evaluation/latency/latency_over_time.png")

# --- Plot 2: steady-state distribution ---
plt.figure(figsize=(8, 5))
plt.hist(steady_state["total_ms"], bins=40, edgecolor="black")
plt.axvline(mean_latency, color="green", linestyle="--", label=f"Mean ({mean_latency:.1f}ms)")
plt.axvline(p95_latency, color="orange", linestyle="--", label=f"p95 ({p95_latency:.1f}ms)")
plt.xlabel("Total latency (ms)")
plt.ylabel("Frame count")
plt.title("Latency Distribution (Steady State) — YOLOv8n on CPU")
plt.legend()
plt.tight_layout()
plt.savefig("evaluation/latency/latency_distribution.png", dpi=150)
print("Saved plot: evaluation/latency/latency_distribution.png")

# --- Correlate latency with number of detections ---
correlation = steady_state["total_ms"].corr(steady_state["detections"])
print(f"\nCorrelation between latency and detection count: {correlation:.3f}")
print("(Closer to +1 means more detections strongly relate to higher latency)")

# Look specifically at the slow stretch you noticed (frames 630-750)
slow_stretch = steady_state[(steady_state["frame"] >= 630) & (steady_state["frame"] <= 750)]
rest = steady_state[(steady_state["frame"] < 630) | (steady_state["frame"] > 750)]

print(f"\n--- Frames 630-750 (the stretch you noticed) ---")
print(f"Avg latency: {slow_stretch['total_ms'].mean():.2f} ms")
print(f"Avg detections: {slow_stretch['detections'].mean():.2f}")

print(f"\n--- Rest of the video ---")
print(f"Avg latency: {rest['total_ms'].mean():.2f} ms")
print(f"Avg detections: {rest['detections'].mean():.2f}")

# Scatter plot: detections vs latency
plt.figure(figsize=(8, 5))
plt.scatter(steady_state["detections"], steady_state["total_ms"], alpha=0.4, s=15)
plt.xlabel("Number of detections in frame")
plt.ylabel("Total latency (ms)")
plt.title("Latency vs. Detection Count — YOLOv8n on CPU")
plt.tight_layout()
plt.savefig("evaluation/latency/latency_vs_detections.png", dpi=150)
print("\nSaved plot: evaluation/latency/latency_vs_detections.png")
