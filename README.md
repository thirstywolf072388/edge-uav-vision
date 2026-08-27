# Edge UAV Vision

An autonomous perception-to-action system: a computer vision pipeline detects
objects in a video feed and, upon confirmed detection, autonomously triggers
a simulated UAV flight mission — takeoff, navigate to the target area, return,
and land — with no manual flight control involved.

Built as a hands-on exploration of edge AI, computer vision, and autonomous
systems engineering, with an emphasis on measuring and reasoning about real
engineering tradeoffs (latency, confidence thresholds, false-positive
suppression) rather than just getting a demo to run once.

## What this demonstrates

- Object detection pipeline (YOLOv8) with measured accuracy behavior and
  known failure modes
- Latency benchmarking: cold-start vs. steady-state, percentile analysis,
  correlation testing against detection count
- False-positive suppression via confidence thresholding and multi-frame
  confirmation, with before/after evidence
- Autonomous flight control via PX4 SITL + MAVSDK, with telemetry-verified
  navigation (not blind timing)
- A closed perception-to-action loop: vision output directly drives flight
  decisions

## Architecture
Video feed (file / eventually live camera)
│
▼
YOLOv8 object detection (confidence-filtered)
│
▼
Multi-frame confirmation (N consecutive qualifying frames)
│
▼
Cooldown gate (prevents repeat triggers on a persistent target)
│
▼
MAVSDK flight command → PX4 SITL / Gazebo simulated drone
│
▼
Autonomous mission: arm → takeoff → navigate to target area →
return to home → land


## Key findings

### Latency (YOLOv8n, CPU, WSL2)

| Metric | Value |
|---|---|
| Cold start (first inference) | 2390 ms (one-time model/graph init cost) |
| Steady-state mean | 9.09 ms |
| Steady-state median | 8.60 ms |
| p95 | 12.71 ms |
| p99 | 16.42 ms |
| Mean FPS | 112.6 |

Cold start is reported separately from steady-state performance since it
reflects a one-time initialization cost, not per-frame inference speed —
conflating the two badly skews any average.

Latency shows only a weak correlation with detection count (r = 0.19),
suggesting per-frame object count is not the primary driver of latency
variance. The most likely explanation is host-system scheduling noise
inherent to running inference on a shared, general-purpose OS via WSL2 —
a hypothesis that becomes testable once the same benchmark runs on
dedicated edge hardware (Jetson) without a shared-resource host OS in
the loop.

### Detection accuracy — stock COCO-pretrained model

Tested against real outdoor/wildlife footage:

- Strong, reliable performance on: people, bears, large birds
- **Deer is not a COCO class** — consistently misclassified as "cow"
  (closest visual analog in the model's known classes)
- Occasional cat → bear misclassification, likely pose/lighting dependent
- Struggles to detect small, fast-moving objects (e.g. hummingbirds),
  likely due to their small pixel footprint per frame — a known
  limitation of general-purpose detectors on tiny objects

This confirms that reliable deer-specific detection would require a
custom-trained model on a labeled deer dataset — a deliberate scope
decision was made to use stock COCO classes for this phase of the
project instead.

### Confidence threshold + multi-frame confirmation

Initial trigger logic (single-frame, conf ≥ 0.5) fired on a borderline
0.53-confidence detection — a real false-positive risk for a system that
commits physical resources (battery, flight time) on every trigger.

Fix: raised confidence threshold to 0.65 and required 5 consecutive
qualifying frames before triggering. Verified behavior:

- Brief low-confidence flickers (single frames around 0.65-0.66) are
  correctly ignored — the confirmation counter resets before reaching
  threshold
- A persistent, high-confidence target correctly triggers only once per
  cooldown window, even while continuously qualifying across hundreds
  of frames — confirmation prevents false triggers, cooldown prevents
  redundant triggers on the same ongoing detection

## Verified autonomous flight

Early flight-mission code relied on fixed `sleep()` timers to assume
mission progress, which produced misleadingly "successful" console output
even when the drone landed immediately after takeoff due to a simulated
sensor timing fault. Rewritten to poll real telemetry (position, in-air
state, flight mode) instead of trusting elapsed time.

Verified result: confirmed GPS-tracked flight from home position to a
50m waypoint and back, with altitude holding steady at the commanded
10m throughout transit.

## Stack

- **Vision**: Python, OpenCV, Ultralytics YOLOv8
- **Flight**: PX4 Autopilot (SITL), Gazebo simulator, MAVSDK
- **Analysis**: pandas, matplotlib
- **Environment**: Ubuntu 22.04 (WSL2)

## Repository structure
edge-uav-vision/
├── vision/
│ ├── preprocessing/ # frame reading, motion detection
│ └── detection/ # YOLO detection + latency instrumentation
├── flight/
│ ├── px4/ # scripted MAVSDK flight (arm/takeoff/waypoint/land)
│ └── autonomy/ # vision-triggered flight logic, confirmation/cooldown gating
├── evaluation/
│ └── latency/ # timing CSVs, analysis script, plots
├── data/ # test video/image assets (gitignored)
└── README.md


## What's next

- Custom-trained detector for classes outside COCO (e.g. deer), using
  the same confidence/confirmation pipeline already in place
- Deploy to real edge hardware (NVIDIA Jetson) and re-run the latency
  benchmark to compare against WSL2/CPU baseline
- Move from simulated PX4 flight to a real flight controller, starting
  with bench testing before any actual flight
- Live camera input in place of recorded video

## Honest limitations

- All flight testing to date is in simulation (PX4 SITL / Gazebo); no
  real hardware has flown yet
- Detection is limited to COCO's 80 classes; deer and similar
  non-standard classes are approximated via the closest available class
- Latency benchmarks reflect CPU-only inference inside WSL2, not representative of real edge-hardware performance
