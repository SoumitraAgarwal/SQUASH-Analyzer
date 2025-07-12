import cv2
import numpy as np
import os

def create_templates_from_video(
    video_path,
    resize_shape=(160, 90),
    diff_threshold=25,
    max_cameras=7,
    no_match_frames=210,  # 7 seconds at 30fps
    frame_stride=5,
    show_progress=False
):
    """
    Create camera templates from a video. Only create a new template if no match is found for a sustained period.
    The template is the average of all frames in the no-match period.
    This function does not write any files; it only returns templates in memory.
    All outputs (if any) must be inside the input video directory or its subfolders.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    templates = []
    no_match_buffer = []
    no_match_count = 0
    frame_idx = 0
    progress_iter = None
    if show_progress:
        from tqdm import tqdm
        progress_iter = tqdm(total=total_frames, desc="Template creation pass")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            if progress_iter is not None:
                progress_iter.update(1)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resize_shape).astype(np.float32)
        # Check match - optimized with early exit
        min_diff = float('inf')
        if templates:
            for template in templates:
                diff = np.mean(np.abs(template - small))
                if diff < min_diff:
                    min_diff = diff
                    if diff < diff_threshold:  # Early exit if good match found
                        break
        if min_diff < diff_threshold and len(templates) < max_cameras:
            # Match found, reset buffer
            no_match_buffer = []
            no_match_count = 0
        else:
            # No match, buffer this frame
            no_match_buffer.append(small)
            no_match_count += 1
            # If enough no-match frames, create new template
            if no_match_count >= no_match_frames and len(templates) < max_cameras:
                avg_template = np.mean(no_match_buffer, axis=0)
                templates.append(avg_template)
                no_match_buffer = []
                no_match_count = 0
        frame_idx += 1
        if progress_iter is not None:
            progress_iter.update(1)
    cap.release()
    if progress_iter is not None:
        progress_iter.close()
    return templates

def assign_frames_to_templates(
    video_path,
    templates,
    resize_shape=(160, 90),
    diff_threshold=25,
    frame_stride=1,
    max_process_time_s=float('inf'),
    output_annotated=None,
    output_debug=None,
    max_cameras=7,
    enable_grid_debug=True,
    show_progress=False
):
    """
    For each frame, assign it to the closest template and optionally write annotated and debug videos.
    If output_annotated or output_debug are provided, they should be paths inside the same directory as the input video or its subfolders.
    This function will not write files outside the camera folder structure.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    CELL_W, CELL_H = 320, 180
    GRID_W, GRID_H = CELL_W * 3, CELL_H * 4
    fourcc = 0x7634706d  # 'mp4v' as integer
    out = None
    grid_out = None
    if output_annotated:
        # Ensure output is inside the input video directory or its subfolders
        if not os.path.abspath(output_annotated).startswith(os.path.dirname(os.path.abspath(video_path))):
            raise ValueError("output_annotated must be inside the input video directory or its subfolders.")
        out = cv2.VideoWriter(output_annotated, fourcc, fps, (width, height))
    if enable_grid_debug and output_debug:
        if not os.path.abspath(output_debug).startswith(os.path.dirname(os.path.abspath(video_path))):
            raise ValueError("output_debug must be inside the input video directory or its subfolders.")
        grid_out = cv2.VideoWriter(output_debug, fourcc, fps, (GRID_W, GRID_H))
    frame_idx = 0
    progress_iter = None
    if show_progress:
        from tqdm import tqdm
        progress_iter = tqdm(total=total_frames, desc="Assignment & debug video pass")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            if progress_iter is not None:
                progress_iter.update(1)
            continue
        if frame_idx / fps >= max_process_time_s:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resize_shape).astype(np.float32)
        # Assign to closest template - optimized
        detected_camera = 0
        if templates:
            min_diff = float('inf')
            for i, template in enumerate(templates):
                diff = np.mean(np.abs(template - small))
                if diff < min_diff:
                    min_diff = diff
                    detected_camera = i
        # Annotate frame
        if out is not None:
            cam_label = f"Camera {detected_camera + 1}"
            frame_annot = frame.copy()
            cv2.putText(frame_annot, cam_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            out.write(frame_annot)
        # Debug video
        if enable_grid_debug and grid_out is not None:
            template_imgs = []
            for i in range(max_cameras):
                if i < len(templates):
                    t = templates[i].astype(np.uint8)
                    t_bgr = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
                    t_bgr = cv2.resize(t_bgr, (CELL_W, CELL_H), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(t_bgr, f"T{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                    template_imgs.append(t_bgr)
                else:
                    template_imgs.append(np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8))
            top_row = np.hstack(template_imgs[:3])
            mid_row = np.hstack(template_imgs[3:6])
            frame_disp = cv2.resize(frame, (CELL_W, CELL_H))
            if detected_camera < len(templates):
                diff_img = np.abs(small - templates[detected_camera])
                diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
                diff_img = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)
                diff_img = cv2.resize(diff_img, (CELL_W, CELL_H))
            else:
                diff_img = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            mid_blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            middle_row = np.hstack([frame_disp, diff_img, mid_blank])
            bottom_row = np.zeros((CELL_H, GRID_W, 3), dtype=np.uint8)
            grid_frame = np.vstack([top_row, mid_row, middle_row, bottom_row])
            grid_frame = grid_frame[:GRID_H, :GRID_W]
            grid_out.write(grid_frame)
        frame_idx += 1
        if progress_iter is not None:
            progress_iter.update(1)
    cap.release()
    if out is not None:
        out.release()
    if grid_out is not None:
        grid_out.release() 