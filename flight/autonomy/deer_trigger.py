from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

# NOTE: standard COCO classes don't include "deer" -- using "horse" as a
# rough stand-in for now so the trigger logic can be built and tested.
# This gets replaced once a custom-trained deer model exists.
TARGET_CLASSES = ["horse", "dog", "cow", "bear"]  # closest proxies for now
COOLDOWN_SECONDS = 30

video_path = "data/vtest.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

last_trigger_time = 0
frame_count = 0

def trigger_alarm(detected_class, confidence):
    """
    Placeholder for the real speaker trigger.
    Later this becomes an HTTP/MQTT call to the ESP32 speaker nodes.
    """
    print(f"🚨 ALARM TRIGGERED — detected '{detected_class}' "
          f"({confidence:.2f} confidence). Sending signal to speakers...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame, verbose=False, conf=0.5)

    for box in results[0].boxes:
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)

        if class_name in TARGET_CLASSES:
            now = time.time()
            time_since_last_trigger = now - last_trigger_time

            if time_since_last_trigger >= COOLDOWN_SECONDS:
                trigger_alarm(class_name, confidence)
                last_trigger_time = now
            else:
                remaining = COOLDOWN_SECONDS - time_since_last_trigger
                print(f"Frame {frame_count}: '{class_name}' detected but "
                      f"in cooldown ({remaining:.1f}s remaining)")

cap.release()
print(f"Done. Processed {frame_count} frames.")
