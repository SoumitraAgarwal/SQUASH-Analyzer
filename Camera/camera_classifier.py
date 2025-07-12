from camera_templates import create_templates_from_video, assign_frames_to_templates

VIDEO_DIR = "WC_2425"
RESIZE_SHAPE = (160, 90)
DIFF_THRESHOLD = 25
MAX_CAMERAS = 7
NO_MATCH_SECONDS = 7
FRAME_STRIDE = 1
ENABLE_GRID_DEBUG = True
MAX_PROCESS_TIME_S = float('inf')

import os

video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
if not video_files:
    raise FileNotFoundError(f"No mp4 files found in {VIDEO_DIR}")
INPUT_VIDEO = os.path.join(VIDEO_DIR, video_files[0])

import cv2
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
NO_MATCH_FRAMES = int(NO_MATCH_SECONDS * fps)

# Create templates
templates = create_templates_from_video(
    INPUT_VIDEO,
    resize_shape=RESIZE_SHAPE,
    diff_threshold=DIFF_THRESHOLD,
    max_cameras=MAX_CAMERAS,
    no_match_frames=NO_MATCH_FRAMES,
    frame_stride=3  # Optimized: use every 3rd frame for template creation
)

# Assign and annotate
assign_frames_to_templates(
    INPUT_VIDEO,
    templates,
    resize_shape=RESIZE_SHAPE,
    diff_threshold=DIFF_THRESHOLD,
    frame_stride=FRAME_STRIDE,
    max_process_time_s=MAX_PROCESS_TIME_S,
    output_annotated="annotated_camera_video.mp4",
    output_debug="camera_grid_debug.mp4" if ENABLE_GRID_DEBUG else None,
    max_cameras=MAX_CAMERAS,
    enable_grid_debug=ENABLE_GRID_DEBUG
)

print("Annotated video saved to annotated_camera_video.mp4")
if ENABLE_GRID_DEBUG:
    print("Grid debug video saved to camera_grid_debug.mp4") 