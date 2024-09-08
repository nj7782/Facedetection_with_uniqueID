import cv2
import numpy as np
import torch
import sys
from sort import Sort  # Import the SORT tracker

# Paths Configuration
yolov5_path = r"C:\Users\jitin\us\yolov5"
sort_path = r"C:\Users\jitin\us\sort"  # Path where SORT is located
weights_path = r"C:\Users\jitin\Downloads\yolov5s.pt"
video_path = r"C:\nj\ABA Therapy - Play.mp4"

# Append SORT library to system path
sys.path.append(sort_path)

# Initialize SORT tracker with adjusted parameters
tracker = Sort(max_age=15, min_hits=5, iou_threshold=0.3)

# Load YOLOv5 model
model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
model.eval()

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file or no frames captured.")
        break

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img)

    # Parse detections
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    # Filter detections for 'person' class (class_id = 0 in COCO)
    person_detections = detections[detections[:, 5] == 0]  # Adjust class_id if needed

    # Further filter based on confidence
    confidence_threshold = 0.5
    person_detections = person_detections[person_detections[:, 4] >= confidence_threshold]

    # Prepare detections for SORT [x1, y1, x2, y2, score]
    sort_detections = person_detections[:, :5]

    # Update SORT tracker
    tracked_objects = tracker.update(sort_detections)

    # Draw bounding boxes and IDs on the frame
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put ID text
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 + SORT Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
