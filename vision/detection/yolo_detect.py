from ultralytics import YOLO
import cv2
import time
import csv

model = YOLO("yolov8n.pt")

video_path = "data/youtube_test.mkv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
timing_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --- Preprocessing stage ---
    t0 = time.perf_counter()
    # (YOLO handles most preprocessing internally, but we mark the boundary anyway)
    t1 = time.perf_counter()

    # --- Inference stage ---
    results = model(frame, verbose=False, conf=0.5)
    t2 = time.perf_counter()

    # Live console output every frame
    classes_found = [model.names[int(box.cls)] for box in results[0].boxes]
    print(f"Frame {frame_count} | {len(classes_found)} detection(s): {classes_found}")

    # Live video window with boxes drawn
    annotated_frame = results[0].plot()
    cv2.imshow("Live Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit requested via 'q' key")
        break

    # --- Postprocessing stage ---
    num_detections = len(results[0].boxes)
    t3 = time.perf_counter()

    preprocess_ms = (t1 - t0) * 1000
    inference_ms = (t2 - t1) * 1000
    postprocess_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000
    fps = 1000 / total_ms if total_ms > 0 else 0

    timing_log.append({
        "frame": frame_count,
        "preprocess_ms": round(preprocess_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "postprocess_ms": round(postprocess_ms, 2),
        "total_ms": round(total_ms, 2),
        "fps": round(fps, 2),
        "detections": num_detections
    })

    if frame_count % 30 == 0:
        out_path = f"data/yolo_frame_{frame_count}.jpg"
        cv2.imwrite(out_path, annotated_frame)
        print(f"Frame {frame_count} | {total_ms:.1f}ms total | {fps:.1f} FPS | {num_detections} detections")
cv2.destroyAllWindows()
cap.release()

# Save full timing log to CSV
csv_path = "evaluation/latency/yolo_cpu_timing.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=timing_log[0].keys())
    writer.writeheader()
    writer.writerows(timing_log)

# Summary stats
avg_total = sum(r["total_ms"] for r in timing_log) / len(timing_log)
avg_fps = sum(r["fps"] for r in timing_log) / len(timing_log)
avg_inference = sum(r["inference_ms"] for r in timing_log) / len(timing_log)

print(f"\n--- Summary ---")
print(f"Frames processed: {frame_count}")
print(f"Avg inference time: {avg_inference:.2f} ms")
print(f"Avg total time/frame: {avg_total:.2f} ms")
print(f"Avg FPS: {avg_fps:.2f}")
print(f"Timing data saved to: {csv_path}")
