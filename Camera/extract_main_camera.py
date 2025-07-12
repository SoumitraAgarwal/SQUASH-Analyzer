import os
import cv2
import numpy as np
from camera_templates import create_templates_from_video, assign_frames_to_templates
from tqdm import tqdm

VIDEO_DIR = "../WC_2425"
OUTPUT_ROOT = "camera_outputs"
SEGMENTS_DIR = os.path.join(OUTPUT_ROOT, "main_camera_segments")
TEMPLATES_JPEG_DIR = os.path.join(OUTPUT_ROOT, "templates_jpeg")
RESIZE_SHAPE = (160, 90)
DIFF_THRESHOLD = 25
MAX_CAMERAS = 7
NO_MATCH_SECONDS = 5  # Reduced from 7 to 5 seconds for faster template detection
TEMPLATE_CREATION_STRIDE = 1  # Process every 3rd frame for template creation
ASSIGNMENT_STRIDE = 1

os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_JPEG_DIR, exist_ok=True)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
if not video_files:
    raise FileNotFoundError(f"No mp4 files found in {VIDEO_DIR}")

with tqdm(total=len(video_files), desc="Processing videos") as overall_pbar:
    for video_file in video_files:
        input_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        no_match_frames = int(NO_MATCH_SECONDS * fps)
        cap.release()

        # 1. Create templates
        templates = create_templates_from_video(
            input_path,
            resize_shape=RESIZE_SHAPE,
            diff_threshold=DIFF_THRESHOLD,
            max_cameras=MAX_CAMERAS,
            no_match_frames=no_match_frames,
            frame_stride=TEMPLATE_CREATION_STRIDE,
            show_progress=False
        )

        # 2. Save templates as a JPEG grid
        CELL_W, CELL_H = 320, 180
        grid_imgs = []
        for i in range(MAX_CAMERAS):
            if i < len(templates):
                t = templates[i].astype(np.uint8)
                t_bgr = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
                t_bgr = cv2.resize(t_bgr, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST)
                cv2.putText(t_bgr, f"T{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                grid_imgs.append(t_bgr)
            else:
                grid_imgs.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
        top_row = np.hstack(grid_imgs[:3])
        mid_row = np.hstack(grid_imgs[3:6])
        bottom_row = np.hstack(grid_imgs[6:7] + [np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)]*2)
        template_grid = np.vstack([top_row, mid_row, bottom_row])
        jpeg_name = os.path.splitext(video_file)[0] + "_templates.jpg"
        jpeg_path = os.path.join(TEMPLATES_JPEG_DIR, jpeg_name)
        cv2.imwrite(jpeg_path, template_grid)

        # 3. Assign each frame to a template and record assignments - optimized
        assignments = []
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        # Use larger stride for assignment to reduce processing time
        assignment_stride = max(ASSIGNMENT_STRIDE, 3)  # Process every 3rd frame minimum
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % assignment_stride != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, RESIZE_SHAPE).astype(np.float32)
            if templates:
                # Optimized template matching with early exit
                min_diff = float('inf')
                best_idx = 0
                for i, template in enumerate(templates):
                    diff = np.mean(np.abs(template - small))
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
                assignments.append(best_idx)
            else:
                assignments.append(0)
            frame_idx += 1
        cap.release()

        # 4. Find the main template (most used)
        assignments_np = np.array(assignments)
        if len(assignments_np) == 0:
            print(f"No frames found for {video_file}, skipping.")
            overall_pbar.update(1)
            continue
        main_template = int(np.bincount(assignments_np).argmax())
        main_count = np.sum(assignments_np == main_template)
        print(f"{video_file}: Main template is {main_template+1} (used {main_count} frames)")

        # 5. Extract and save main camera segment - optimized
        cap = cv2.VideoCapture(input_path)
        output_path = os.path.join(SEGMENTS_DIR, video_file)
        fourcc = 0x7634706d  # 'mp4v' as integer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_idx = 0
        assign_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % assignment_stride == 0:  # Use same stride as assignment
                if assign_idx < len(assignments_np) and assignments_np[assign_idx] == main_template:
                    out.write(frame)
                assign_idx += 1
            frame_idx += 1
        cap.release()
        out.release()
        print(f"Saved main camera segment for {video_file} to {output_path}")
        print(f"Saved template JPEG for {video_file} to {jpeg_path}")
        overall_pbar.update(1) 