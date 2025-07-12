import cv2
import os
import numpy as np
from tqdm import tqdm

# Directory containing the videos
VIDEO_DIR = "/Users/soumitraagarwal/Squash/WC_2425"
OUTPUT_DIR = "annotated_videos_ball"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all mp4 files in the directory and pick the first one
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
if not video_files:
    raise FileNotFoundError(f"No mp4 files found in {VIDEO_DIR}")
INPUT_VIDEO = os.path.join(VIDEO_DIR, video_files[0])
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_VIDEO))

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

progress_interval = max(1, total_frames // 20)  # 5% intervals
next_progress = progress_interval

prev_gray = None

# Blob detector params (OpenCV 3+)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 10
params.maxArea = 200
params.filterByCircularity = True
params.minCircularity = 0.5
if hasattr(cv2, 'SimpleBlobDetector_create'):
    detector = cv2.SimpleBlobDetector_create(params)
else:
    detector = cv2.SimpleBlobDetector(params)

with tqdm(total=total_frames, desc="Annotating ball") as pbar:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Ball detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ball_center = None
        if prev_gray is not None:
            # Motion mask
            diff = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

            # Color mask for black ball
            _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            # Color mask for white ball
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Combine masks with motion
            black_combined = cv2.bitwise_and(motion_mask, black_mask)
            white_combined = cv2.bitwise_and(motion_mask, white_mask)

            # Detect black ball
            keypoints_black = detector.detect(black_combined) if black_combined is not None else []
            # Detect white ball
            keypoints_white = detector.detect(white_combined) if white_combined is not None else []

            # Pick the largest blob among both masks (if any)
            all_keypoints = list(keypoints_black) + list(keypoints_white)
            if all_keypoints:
                all_keypoints = sorted(all_keypoints, key=lambda k: k.size, reverse=True)
                k = all_keypoints[0]
                x, y = int(k.pt[0]), int(k.pt[1])
                ball_center = (x, y)
                cv2.circle(frame, ball_center, int(k.size // 2), (0, 0, 255), 2)
                cv2.putText(frame, "Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        prev_gray = gray.copy()

        out.write(frame)
        pbar.update(1)
        frame_idx += 1
        if frame_idx >= next_progress:
            percent = int((frame_idx / total_frames) * 100)
            print(f"At {percent}%: Ball: {ball_center}")
            next_progress += progress_interval

cap.release()
out.release()
print(f"Annotated video saved to {OUTPUT_VIDEO}") 