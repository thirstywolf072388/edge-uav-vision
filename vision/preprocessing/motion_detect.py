import cv2

video_path = "data/vtest.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Background subtractor: learns the static background, flags what changes
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Get a mask: white = motion, black = static background
    fg_mask = back_sub.apply(frame)

    # Clean up noise in the mask a bit
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Find contours (outlines) of the moving blobs
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections_this_frame = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # ignore tiny noise blobs
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detections_this_frame += 1

    cv2.putText(frame, f"Frame {frame_count} | Moving objects: {detections_this_frame}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if frame_count % 30 == 0:
        out_path = f"data/motion_frame_{frame_count}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path} | {detections_this_frame} moving object(s) detected")

cap.release()
print(f"Done. Total frames processed: {frame_count}")
