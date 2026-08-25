from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

TARGET_CLASSES = ["horse", "dog", "cow", "bear"]
COOLDOWN_SECONDS = 30
last_trigger_time = 0

def trigger_alarm(detected_class, confidence):
    print(f"🚨 ALARM TRIGGERED — detected '{detected_class}' "
          f"({confidence:.2f} confidence). Sending signal to speakers...")

image_path = "data/deer_test.jpg"
results = model(image_path, conf=0.5)

for box in results[0].boxes:
    class_name = model.names[int(box.cls)]
    confidence = float(box.conf)
    print(f"Detected: {class_name} ({confidence:.2f})")

    if class_name in TARGET_CLASSES:
        now = time.time()
        if now - last_trigger_time >= COOLDOWN_SECONDS:
            trigger_alarm(class_name, confidence)
            last_trigger_time = now

# Save annotated result so you can look at it
annotated = results[0].plot()
import cv2
cv2.imwrite("data/horse_test_annotated.jpg", annotated)
print("Saved: data/horse_test_annotated.jpg")
