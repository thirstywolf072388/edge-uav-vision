from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model (nano version - smallest/fastest, good for testing)
# This will auto-download the model weights the first time you run it (~6MB)
model = YOLO("yolov8n.pt")

video_path = "data/vtest.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO inference on this frame
    results = model(frame, verbose=False, conf=0.5)

    # results[0].plot() draws all the boxes/labels/confidence scores automatically
    annotated_frame = results[0].plot()

    # Count detections and print what was found
    num_detections = len(results[0].boxes)

    if frame_count % 30 == 0:
        out_path = f"data/yolo_frame_{frame_count}.jpg"
        cv2.imwrite(out_path, annotated_frame)

        # Print what classes were detected
        classes_found = [model.names[int(box.cls)] for box in results[0].boxes]
        print(f"Saved {out_path} | {num_detections} detection(s): {classes_found}")

cap.release()
print(f"Done. Total frames processed: {frame_count}")
