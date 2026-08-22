import cv2

# Open the video file
video_path = "data/vtest.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()  # ret = success flag, frame = the actual image
    if not ret:
        break  # no more frames, video ended

    frame_count += 1

    # Draw a rectangle on the frame (just to prove we can modify it)
    cv2.rectangle(frame, (50, 50), (200, 150), (0, 255, 0), 2)

    # Put some text on the frame
    cv2.putText(frame, f"Frame {frame_count}", (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save every 30th frame as an image so we can look at it
    if frame_count % 30 == 0:
        out_path = f"data/frame_{frame_count}.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

cap.release()
print(f"Done. Total frames processed: {frame_count}")
