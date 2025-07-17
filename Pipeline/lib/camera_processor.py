"""
Camera Processing Library
Handles main camera extraction and video processing
"""

import os
import sys
import cv2
import numpy as np

# Camera processing constants
RESIZE_SHAPE = (640, 360)
DIFF_THRESHOLD = 0.15
MAX_CAMERAS = 10
NO_MATCH_SECONDS = 60
TEMPLATE_CREATION_STRIDE = 5
ASSIGNMENT_STRIDE = 5

def create_templates_from_video(input_path, templates_output_path, camera_segments_path, max_duration=None, force_reprocess=False):
    """
    Proper camera template detection and main camera extraction
    Uses the working camera template system with time limit
    """
    try:
        # Import the working camera template functions
        import sys
        sys.path.append('/Users/soumitraagarwal/Squash')
        from camera_templates import create_templates_from_video as create_templates_original, assign_frames_to_templates
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}.f399.mp4"
        output_path = os.path.join(camera_segments_path, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(camera_segments_path, exist_ok=True)
        
        # Calculate max process time (use max_duration or default to 90 minutes)
        max_process_time = max_duration if max_duration else 5400  # 90 minutes default
        
        print(f"📹 Creating camera templates (max {max_process_time}s)...")
        
        # Overwrite output if force_reprocess
        if force_reprocess and os.path.exists(output_path):
            os.remove(output_path)
        
        # Create camera templates with time limit and debug video
        templates = create_templates_with_time_limit_and_debug(
            input_path,
            max_process_time,
            resize_shape=(160, 90),  # Reduced from 320x180 to 160x90 for faster processing
            diff_threshold=15,  # Changed back from 10 to 15
            max_cameras=7,
            no_match_frames=210,  # Original working value
            frame_stride=15,  # Increased from 5 to 15 for faster processing
            show_progress=True,
            debug_video_path=os.path.join(templates_output_path, f"{base_name}_debug.mp4"),
            use_top_half=True  # Use only top half of video for template detection
        )
        
        if not templates:
            print("⚠️ No camera templates found, keeping full video")
            import shutil
            shutil.copy2(input_path, output_path)
            return True
            
        print(f"🎯 Found {len(templates)} camera templates")
        
        # Save templates to disk for debugging/reuse
        save_templates_to_disk(templates, templates_output_path, base_name)
        
        # Find the main camera template (usually the one with most frames)
        main_camera_segments = find_main_camera_segments(input_path, templates, max_process_time)
        
        if main_camera_segments:
            success = create_main_camera_video(input_path, output_path, main_camera_segments)
            if success:
                print(f"✅ Main camera extraction: {output_filename}")
            else:
                print("❌ Main camera video creation failed, keeping full video")
                import shutil
                shutil.copy2(input_path, output_path)
        else:
            print("⚠️ No main camera segments found, keeping full video")
            import shutil
            shutil.copy2(input_path, output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Camera processing failed: {e}")
        return False

def find_content_segments(video_path):
    """Fast content detection - samples every 30 seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 30 seconds
        sample_interval = int(30 * fps)
        content_frames = []
        
        print(f"🔍 Sampling video every 30 seconds...")
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Simple content detection - not black screen, has reasonable brightness
            if is_content_frame(frame):
                content_frames.append(frame_idx)
        
        cap.release()
        
        if len(content_frames) < 2:
            return []
        
        # Create segments from first to last content frame
        segments = [{
            'start_frame': max(0, content_frames[0] - int(60 * fps)),  # Start 1 min before first content
            'end_frame': min(total_frames, content_frames[-1] + int(60 * fps)),  # End 1 min after last content
            'start_time': max(0, (content_frames[0] - int(60 * fps)) / fps),
            'end_time': min(total_frames / fps, (content_frames[-1] + int(60 * fps)) / fps)
        }]
        
        duration = segments[0]['end_time'] - segments[0]['start_time']
        print(f"📊 Content segment: {segments[0]['start_time']:.1f}s to {segments[0]['end_time']:.1f}s ({duration:.1f}s)")
        
        return segments
        
    except Exception as e:
        print(f"❌ Content detection failed: {e}")
        return []

def is_content_frame(frame):
    """Fast check if frame has actual content (not black screen, static image)"""
    try:
        # Convert to grayscale for faster analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if frame is mostly black (common in stream padding)
        mean_brightness = np.mean(gray)
        if mean_brightness < 20:  # Very dark frame
            return False
        
        # Check if frame has reasonable contrast (not static image)
        contrast = np.std(gray)
        if contrast < 10:  # Very low contrast (static image)
            return False
        
        # Check if frame has reasonable brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Avoid frames that are too uniform in brightness
        max_bin = np.max(hist)
        if max_bin > len(gray.flatten()) * 0.8:  # 80% of pixels same brightness
            return False
        
        return True
        
    except Exception as e:
        return False

def create_segmented_video(input_path, output_path, segments):
    """Create video with only the content segments"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to create segmented video")
            return False
        
        frames_written = 0
        
        for i, segment in enumerate(segments):
            print(f"📹 Processing segment {i+1}/{len(segments)}: {segment['end_time'] - segment['start_time']:.1f}s")
            
            # Jump to segment start
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start_frame'])
            
            # Copy frames from this segment
            for frame_idx in range(segment['start_frame'], segment['end_frame']):
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
        
        cap.release()
        out.release()
        
        duration = frames_written / fps
        print(f"✅ Segmented video created: {duration:.1f}s ({frames_written} frames)")
        return True
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return False

def create_templates_with_time_limit_and_debug(input_path, max_process_time_s, debug_video_path=None, **kwargs):
    """Create templates with time limit and optional debug video showing the process"""
    try:
        resize_shape = kwargs.get('resize_shape', (320, 180))
        diff_threshold = kwargs.get('diff_threshold', 25)
        max_cameras = kwargs.get('max_cameras', 7)
        no_match_frames = kwargs.get('no_match_frames', 210)
        frame_stride = kwargs.get('frame_stride', 5)
        show_progress = kwargs.get('show_progress', True)
        use_top_half = kwargs.get('use_top_half', False)
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        # Calculate maximum frames to process
        max_frames = min(total_frames, int(max_process_time_s * fps))
        
        print(f"📊 Video: {total_duration:.1f}s, analyzing first {max_process_time_s:.1f}s ({max_frames} frames)")
        if use_top_half:
            print(f"📊 Using top half of video for template detection")
        
        # Debug video setup
        debug_out = None
        CELL_W, CELL_H = 320, 180
        GRID_W, GRID_H = CELL_W * 3, CELL_H * 3
        
        if debug_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            debug_out = cv2.VideoWriter(debug_video_path, fourcc, fps, (GRID_W, GRID_H))
            print(f"📹 Creating debug video: {debug_video_path}")
        
        templates = []
        no_match_buffer = []
        no_match_count = 0
        frame_idx = 0
        
        progress_iter = None
        if show_progress:
            from tqdm import tqdm
            progress_iter = tqdm(total=max_frames, desc="📹 Creating templates (with debug)")
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                if progress_iter is not None:
                    progress_iter.update(1)
                continue
                
            # Crop to top half if requested
            if use_top_half:
                frame_height = frame.shape[0]
                frame = frame[:frame_height//2, :]  # Keep only top half
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, resize_shape).astype(np.float32)
            
            # Check match - optimized with early exit
            min_diff = float('inf')
            matched_template = False
            best_template_idx = -1
            
            if templates:
                for i, template in enumerate(templates):
                    diff = np.mean(np.abs(template - small))
                    if diff < min_diff:
                        min_diff = diff
                        best_template_idx = i
                        if diff < diff_threshold:  # Early exit if good match found
                            matched_template = True
                            break
            
            if matched_template:
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
                    print(f"   📹 Created template {len(templates)} (diff: {min_diff:.1f})")
                    no_match_buffer = []
                    no_match_count = 0
            
            # Create debug frame
            if debug_out:
                debug_frame = create_debug_grid_frame(
                    frame, templates, small, best_template_idx, 
                    matched_template, min_diff, len(templates),
                    CELL_W, CELL_H, GRID_W, GRID_H
                )
                debug_out.write(debug_frame)
            
            frame_idx += 1
            if progress_iter is not None:
                progress_iter.update(1)
        
        cap.release()
        if debug_out:
            debug_out.release()
        if progress_iter is not None:
            progress_iter.close()
        
        print(f"📊 Template creation complete: {len(templates)} templates found")
        if len(templates) < 2:
            print(f"⚠️ Warning: Only {len(templates)} template(s) found. This might indicate:")
            print(f"   - Video has very stable camera angles")
            print(f"   - Diff threshold ({diff_threshold}) might be too low")
            print(f"   - Not enough camera switches in the analyzed duration")
        
        return templates
        
    except Exception as e:
        print(f"❌ Template creation with time limit failed: {e}")
        return []

def create_debug_grid_frame(frame, templates, current_small, best_template_idx, matched, min_diff, num_templates, CELL_W, CELL_H, GRID_W, GRID_H):
    """Create a 3x3 debug grid frame showing current frame and templates"""
    try:
        # Create 3x3 grid
        grid_frame = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
        
        # Resize current frame to fit in cell
        frame_resized = cv2.resize(frame, (CELL_W, CELL_H))
        
        # Add current frame info
        status_text = f"Match: {matched} | Diff: {min_diff:.1f} | Templates: {num_templates}"
        cv2.putText(frame_resized, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Place current frame in center (position 1, 1)
        y_start = CELL_H
        y_end = y_start + CELL_H
        x_start = CELL_W
        x_end = x_start + CELL_W
        grid_frame[y_start:y_end, x_start:x_end] = frame_resized
        
        # Add templates to other cells
        template_positions = [
            (0, 0), (0, 1), (0, 2),  # Top row
            (1, 0),         (1, 2),  # Middle row (1,1 is current frame)
            (2, 0), (2, 1), (2, 2)   # Bottom row
        ]
        
        for i, (row, col) in enumerate(template_positions):
            y_start = row * CELL_H
            y_end = y_start + CELL_H
            x_start = col * CELL_W
            x_end = x_start + CELL_W
            
            if i < len(templates):
                # Convert template to BGR
                template_bgr = cv2.cvtColor(templates[i].astype(np.uint8), cv2.COLOR_GRAY2BGR)
                template_resized = cv2.resize(template_bgr, (CELL_W, CELL_H))
                
                # Highlight if this is the best match
                if i == best_template_idx:
                    cv2.rectangle(template_resized, (0, 0), (CELL_W-1, CELL_H-1), (0, 255, 0), 3)
                
                # Add template info
                cv2.putText(template_resized, f"T{i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                grid_frame[y_start:y_end, x_start:x_end] = template_resized
            else:
                # Empty cell - black with label
                empty_cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
                cv2.putText(empty_cell, "Empty", (CELL_W//2-30, CELL_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                grid_frame[y_start:y_end, x_start:x_end] = empty_cell
        
        return grid_frame
        
    except Exception as e:
        # Return black frame if error
        return np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)

def create_templates_with_time_limit(input_path, max_process_time_s, **kwargs):
    """Create templates with time limit by processing only first N seconds"""
    try:
        resize_shape = kwargs.get('resize_shape', (320, 180))
        diff_threshold = kwargs.get('diff_threshold', 25)
        max_cameras = kwargs.get('max_cameras', 7)
        no_match_frames = kwargs.get('no_match_frames', 210)
        frame_stride = kwargs.get('frame_stride', 5)
        show_progress = kwargs.get('show_progress', True)
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        # Calculate maximum frames to process
        max_frames = min(total_frames, int(max_process_time_s * fps))
        
        print(f"📊 Video: {total_duration:.1f}s, analyzing first {max_process_time_s:.1f}s ({max_frames} frames)")
        
        templates = []
        no_match_buffer = []
        no_match_count = 0
        frame_idx = 0
        
        progress_iter = None
        if show_progress:
            from tqdm import tqdm
            progress_iter = tqdm(total=max_frames, desc="📹 Creating templates")
        
        while frame_idx < max_frames:
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
            matched_template = False
            if templates:
                for template in templates:
                    diff = np.mean(np.abs(template - small))
                    if diff < min_diff:
                        min_diff = diff
                        if diff < diff_threshold:  # Early exit if good match found
                            matched_template = True
                            break
            
            if matched_template:
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
                    print(f"   📹 Created template {len(templates)} (diff: {min_diff:.1f})")
                    no_match_buffer = []
                    no_match_count = 0
            
            frame_idx += 1
            if progress_iter is not None:
                progress_iter.update(1)
        
        cap.release()
        if progress_iter is not None:
            progress_iter.close()
        
        print(f"📊 Template creation complete: {len(templates)} templates found")
        if len(templates) < 2:
            print(f"⚠️ Warning: Only {len(templates)} template(s) found. This might indicate:")
            print(f"   - Video has very stable camera angles")
            print(f"   - Diff threshold ({diff_threshold}) might be too low")
            print(f"   - Not enough camera switches in the analyzed duration")
        
        return templates
        
    except Exception as e:
        print(f"❌ Template creation with time limit failed: {e}")
        return []

def save_templates_to_disk(templates, templates_output_path, base_name):
    """Save camera templates to disk as a 3x3 grid image"""
    try:
        import cv2
        import numpy as np
        
        os.makedirs(templates_output_path, exist_ok=True)
        
        if len(templates) == 0:
            print("⚠️ No templates to save")
            return
        
        # Create a 3x3 grid of templates
        rows, cols = 3, 3
        max_templates = rows * cols
        
        # Get template dimensions
        template_h, template_w = templates[0].shape
        
        # Create larger grid image with padding
        padding = 20
        cell_w = template_w + padding
        cell_h = template_h + padding
        grid_w = cols * cell_w
        grid_h = rows * cell_h
        
        # Create black background
        grid_img = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        # Place templates in grid
        for i in range(min(len(templates), max_templates)):
            row = i // cols
            col = i % cols
            
            # Calculate position with padding
            y_start = row * cell_h + padding // 2
            y_end = y_start + template_h
            x_start = col * cell_w + padding // 2
            x_end = x_start + template_w
            
            # Place template
            grid_img[y_start:y_end, x_start:x_end] = templates[i].astype(np.uint8)
            
            # Add label
            cv2.putText(grid_img, f"T{i+1}", (x_start + 10, y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Save grid image
        grid_filename = f"{base_name}_templates_3x3_grid.jpg"
        grid_path = os.path.join(templates_output_path, grid_filename)
        cv2.imwrite(grid_path, grid_img)
        
        print(f"📊 Saved {len(templates)} templates in 3x3 grid: {grid_filename}")
        
    except Exception as e:
        print(f"⚠️ Failed to save templates: {e}")

def find_main_camera_segments(video_path, templates, max_process_time_s=5400):
    """Find segments belonging to the main camera template"""
    try:
        print(f"🔍 Analyzing frames to find main camera segments (max {max_process_time_s}s)...")
        
        # Analyze video to assign frames to templates
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames based on time limit
        max_frames = min(total_frames, int(max_process_time_s * fps))
        
        # Count frames for each template
        template_frame_counts = [0] * len(templates)
        frame_assignments = []
        
        frame_idx = 0
        resize_shape = (160, 90)  # Match the resolution used in template creation
        
        # Add progress bar
        from tqdm import tqdm
        progress_bar = tqdm(total=max_frames, desc="🔍 Finding main camera segments")
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames for performance (analyze every 10th frame)
            if frame_idx % 10 != 0:
                frame_idx += 1
                continue
                
            # Crop to top half for consistency with template creation
            frame_height = frame.shape[0]
            frame_cropped = frame[:frame_height//2, :]  # Keep only top half
            
            # Convert and resize frame
            gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, resize_shape).astype(np.float32)
            
            # Find closest template
            best_template = 0
            min_diff = float('inf')
            
            for i, template in enumerate(templates):
                diff = np.mean(np.abs(template - small))
                if diff < min_diff:
                    min_diff = diff
                    best_template = i
            
            # Only count as assigned if it's a good match (same threshold as template creation)
            if min_diff < 15:  # Use same threshold as template creation
                template_frame_counts[best_template] += 1
                frame_assignments.append((frame_idx, best_template, True))  # True = good match
            else:
                frame_assignments.append((frame_idx, -1, False))  # -1 = no good match
            frame_idx += 1
            progress_bar.update(1)
        
        cap.release()
        progress_bar.close()
        
        # Find the main camera (template with most frames)
        main_camera_idx = np.argmax(template_frame_counts)
        main_camera_frame_count = template_frame_counts[main_camera_idx]
        
        print(f"🎯 Template frame counts: {template_frame_counts}")
        print(f"📹 Main camera: Template {main_camera_idx + 1} ({main_camera_frame_count} frames)")
        
        # Create segments for main camera (only include frames with good matches)
        segments = []
        current_segment_start = None
        min_segment_duration = 15.0  # 15 seconds minimum (increased from 5)
        
        for frame_idx, template_idx, is_good_match in frame_assignments:
            # Only include frames that have good matches AND belong to main camera
            if is_good_match and template_idx == main_camera_idx:
                if current_segment_start is None:
                    current_segment_start = frame_idx
            else:
                if current_segment_start is not None:
                    # End current segment
                    duration = (frame_idx - current_segment_start) / fps
                    if duration >= min_segment_duration:
                        segments.append({
                            'start_frame': current_segment_start,
                            'end_frame': frame_idx,
                            'start_time': current_segment_start / fps,
                            'end_time': frame_idx / fps,
                            'duration': duration
                        })
                    current_segment_start = None
        
        # Handle final segment
        if current_segment_start is not None:
            duration = (max_frames - current_segment_start) / fps
            if duration >= min_segment_duration:
                segments.append({
                    'start_frame': current_segment_start,
                    'end_frame': max_frames,
                    'start_time': current_segment_start / fps,
                    'end_time': max_frames / fps,
                    'duration': duration
                })
        
        total_duration = sum(s['duration'] for s in segments)
        total_frames_analyzed = len(frame_assignments)
        good_matches = sum(1 for _, _, is_good_match in frame_assignments if is_good_match)
        
        print(f"📊 Found {len(segments)} main camera segments, total duration: {total_duration:.1f}s")
        print(f"📊 Frame analysis: {good_matches}/{total_frames_analyzed} frames had good matches ({good_matches/total_frames_analyzed*100:.1f}%)")
        
        return segments
        
    except Exception as e:
        print(f"❌ Main camera detection failed: {e}")
        return []

def create_main_camera_video(input_path, output_path, segments):
    """Create video with only the main camera segments"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("❌ Failed to create main camera video")
            return False
        
        frames_written = 0
        
        for i, segment in enumerate(segments):
            print(f"📹 Processing main camera segment {i+1}/{len(segments)}: {segment['duration']:.1f}s")
            
            # Jump to segment start
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start_frame'])
            
            # Copy frames from this segment
            for frame_idx in range(segment['start_frame'], segment['end_frame']):
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
        
        cap.release()
        out.release()
        
        duration = frames_written / fps
        print(f"✅ Main camera video created: {duration:.1f}s ({frames_written} frames)")
        return True
        
    except Exception as e:
        print(f"❌ Main camera video creation failed: {e}")
        return False

class CameraProcessor:
    """Handles camera extraction and main camera detection"""
    
    def __init__(self):
        pass
    
    def extract_main_camera_segments(self, input_path, output_dir, max_duration=None, force_reprocess=False):
        """Extract main camera segments from video"""
        try:
            print(f"📹 Processing camera extraction for: {os.path.basename(input_path)}")
            
            # Check if video exists
            if not os.path.exists(input_path):
                print(f"❌ Video file not found: {input_path}")
                return None
            
            # Create templates
            templates_output_path = os.path.join(output_dir, "templates")
            os.makedirs(templates_output_path, exist_ok=True)
            
            camera_segments_path = os.path.join(output_dir, "main_camera_segments")
            os.makedirs(camera_segments_path, exist_ok=True)
            
            # Extract templates and main camera
            success = create_templates_from_video(
                input_path,
                templates_output_path,
                camera_segments_path,
                max_duration=max_duration,
                force_reprocess=force_reprocess
            )
            
            if success:
                # The output file should be created directly in camera_segments_path
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_filename = f"{base_name}.f399.mp4"
                expected_output_path = os.path.join(camera_segments_path, output_filename)
                
                if os.path.exists(expected_output_path):
                    print(f"✅ Camera extraction successful: {output_filename}")
                    return expected_output_path
                else:
                    print(f"❌ Main camera file not found: {expected_output_path}")
                    return None
            else:
                print(f"❌ Camera extraction failed")
                return None
                
        except Exception as e:
            print(f"❌ Camera extraction error: {e}")
            return None
    
    def get_video_info(self, video_path):
        """Get basic video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return None