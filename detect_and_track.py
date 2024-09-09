import cv2
import numpy as np
import torch
import sys
import time
from sort import Sort  # Import the SORT tracker

# Paths Configuration
yolov5_path = r"C:\Users\jitin\us\yolov5"
sort_path = r"C:\Users\jitin\us\sort"  # Path where SORT is located
weights_path = r"C:\Users\jitin\Downloads\yolov5s.pt"
video_path = r"C:\nj\ABA Therapy - Play.mp4"
output_folder = r"C:\nj\saved_faces"  # Folder to save cropped face images

# Append SORT library to system path
sys.path.append(sort_path)

# Initialize SORT tracker with adjusted parameters
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.4)  # Increased IoU and max_age

# Load YOLOv5 model
model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
model.eval()

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    sys.exit()

# Initialize variables
last_save_time = 0
save_interval = 10  # Save faces every 10 seconds
confidence_threshold = 0.4  # Lower confidence threshold for more detections

# Function to save face images
def save_face(image, x1, y1, x2, y2, track_id):
    face_img = image[y1:y2, x1:x2]
    file_name = f"{output_folder}/face_{track_id}_{time.time()}.jpg"
    cv2.imwrite(file_name, face_img)
    print(f"Saved face for ID {track_id} at {file_name}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file or no frames captured.")
        break

    # Get current time
    current_time = time.time()

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img)

    # Parse detections
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    # Filter detections for 'person' class (class_id = 0 in COCO)
    person_detections = detections[detections[:, 5] == 0]  # Adjust class_id if needed

    # Further filter based on confidence
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

        # Save faces every 10 seconds
        if current_time - last_save_time >= save_interval:
            save_face(frame, x1, y1, x2, y2, track_id)

    # Update last save time after saving
    if current_time - last_save_time >= save_interval:
        last_save_time = current_time

    # Display the resulting frame
    cv2.imshow('YOLOv5 + SORT Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
