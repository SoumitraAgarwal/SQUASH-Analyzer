import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Directory containing the videos
VIDEO_DIR = "/Users/soumitraagarwal/Squash/WC_2425"
OUTPUT_DIR = "annotated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all mp4 files in the directory and pick the first one
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
if not video_files:
    raise FileNotFoundError(f"No mp4 files found in {VIDEO_DIR}")
INPUT_VIDEO = os.path.join(VIDEO_DIR, video_files[0])

# Open video to get dimensions
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_margin = int(0.10 * width)
cropped_width = width - 2 * crop_margin
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_VIDEO))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (cropped_width, height))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load YOLOv8 model (pretrained on COCO, class 0 is 'person')
model = YOLO('yolov8n.pt')

progress_interval = max(1, total_frames // 20)  # 5% intervals
next_progress = progress_interval

# For player continuity
prev_centroids = None  # [(x, y), (x, y)]

with tqdm(total=total_frames, desc="Annotating video") as pbar:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop 10% from both sides
        cropped_frame = frame[:, crop_margin:width-crop_margin]

        # Player detection
        results = model(cropped_frame, verbose=False)
        player_boxes = []
        centroids = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # 0 is 'person' in COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    player_boxes.append((x1, y1, x2, y2))
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centroids.append((cx, cy))

        labels = ["", ""]
        if len(centroids) == 2:
            # If previous centroids exist, match by minimum distance
            if prev_centroids is not None and len(prev_centroids) == 2:
                dists = np.zeros((2, 2))
                for i, c in enumerate(centroids):
                    for j, pc in enumerate(prev_centroids):
                        dists[i, j] = np.hypot(c[0] - pc[0], c[1] - pc[1])
                # Hungarian algorithm for optimal assignment (since only 2, just check both)
                if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0]:
                    labels = ["Player 1", "Player 2"]
                    prev_centroids = [centroids[0], centroids[1]]
                else:
                    labels = ["Player 2", "Player 1"]
                    prev_centroids = [centroids[1], centroids[0]]
            else:
                # Assign leftmost as Player 1, rightmost as Player 2
                idxs = np.argsort([c[0] for c in centroids])
                labels = ["", ""]
                labels[idxs[0]] = "Player 1"
                labels[idxs[1]] = "Player 2"
                prev_centroids = [centroids[idxs[0]], centroids[idxs[1]]]

        # Draw boxes and labels if exactly two players
        if len(centroids) == 2:
            for (x1, y1, x2, y2), label in zip(player_boxes, labels):
                label_str = label if label is not None else "Player"
                cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cropped_frame, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        out.write(cropped_frame)
        pbar.update(1)
        frame_idx += 1
        if frame_idx >= next_progress:
            percent = int((frame_idx / total_frames) * 100)
            print(f"At {percent}%: Detected {len(player_boxes)} players. Example boxes: {player_boxes}")
            next_progress += progress_interval

cap.release()
out.release()
print(f"Annotated video saved to {OUTPUT_VIDEO}")
