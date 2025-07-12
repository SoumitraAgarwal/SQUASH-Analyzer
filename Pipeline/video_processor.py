import os
import sys
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import json
import shutil

# Add parent directories to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Camera'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Players'))

@dataclass
class ProcessingTask:
    """Represents a video processing task."""
    video_id: str
    video_title: str
    playlist_title: str
    input_path: str
    stage: str = "pending"  # pending, camera, player, completed, failed
    camera_output: Optional[str] = None
    player_output: Optional[str] = None
    error_message: Optional[str] = None

class VideoProcessor:
    """
    Handles the processing pipeline: Camera extraction → Player detection
    with parallel processing and progress tracking.
    """
    
    def __init__(self, base_output_dir="Pipeline_Output", max_workers=2):
        self.base_output_dir = base_output_dir
        self.max_workers = max_workers
        
        # Folder structure
        self.folders = {
            'downloads': os.path.join(base_output_dir, "01_Downloads"),
            'camera_segments': os.path.join(base_output_dir, "02_Camera_Segments"),
            'player_annotations': os.path.join(base_output_dir, "03_Player_Annotations"),
            'final_outputs': os.path.join(base_output_dir, "04_Final_Outputs"),
            'logs': os.path.join(base_output_dir, "logs")
        }
        
        # Ensure all directories exist
        for folder in self.folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Processing queues
        self.camera_queue = queue.Queue()
        self.player_queue = queue.Queue()
        self.completed_queue = queue.Queue()
        
        # Progress tracking
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        
        # Threading control
        self.stop_event = threading.Event()
        
        print(f"🎬 Video processor initialized with {max_workers} workers")
    
    def add_video_for_processing(self, video_info):
        """Add a video to the processing queue."""
        task = ProcessingTask(
            video_id=video_info.id,
            video_title=video_info.title,
            playlist_title=video_info.playlist_title,
            input_path=video_info.filepath
        )
        
        self.camera_queue.put(task)
        self.total_videos += 1
        print(f"📝 Added to processing queue: {video_info.title}")
    
    def process_camera_extraction(self, task: ProcessingTask) -> ProcessingTask:
        """Extract main camera segments from video."""
        if self.stop_event.is_set():
            task.stage = "cancelled"
            return task
        
        task.stage = "camera"
        
        try:
            # Import camera extraction modules
            from Camera.extract_main_camera import (
                create_templates_from_video, 
                VIDEO_DIR, OUTPUT_ROOT, SEGMENTS_DIR, TEMPLATES_JPEG_DIR,
                RESIZE_SHAPE, DIFF_THRESHOLD, MAX_CAMERAS, NO_MATCH_SECONDS,
                TEMPLATE_CREATION_STRIDE, ASSIGNMENT_STRIDE
            )
            import cv2
            import numpy as np
            
            # Setup output directories for this video
            video_output_dir = os.path.join(self.folders['camera_segments'], task.playlist_title)
            segments_dir = os.path.join(video_output_dir, "main_camera_segments")
            templates_dir = os.path.join(video_output_dir, "templates_jpeg")
            
            os.makedirs(segments_dir, exist_ok=True)
            os.makedirs(templates_dir, exist_ok=True)
            
            # Process the video
            input_path = task.input_path
            
            # Open video to get properties
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {input_path}")
            
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
            
            # 2. Save templates as JPEG grid
            if templates:
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
                
                template_filename = os.path.splitext(os.path.basename(input_path))[0] + "_templates.jpg"
                template_path = os.path.join(templates_dir, template_filename)
                cv2.imwrite(template_path, template_grid)
            
            # 3. Assign frames and find main camera
            assignments = []
            cap = cv2.VideoCapture(input_path)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % ASSIGNMENT_STRIDE != 0:
                    frame_idx += 1
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, RESIZE_SHAPE).astype(np.float32)
                
                if templates:
                    # Optimized template matching
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
            
            # 4. Find main template
            assignments_np = np.array(assignments)
            if len(assignments_np) == 0:
                raise Exception("No frames processed")
            
            main_template = int(np.bincount(assignments_np).argmax())
            
            # 5. Extract main camera segment
            output_filename = os.path.basename(input_path)
            output_path = os.path.join(segments_dir, output_filename)
            
            cap = cv2.VideoCapture(input_path)
            
            # Robust video writer initialization
            fourcc_options = [
                cv2.VideoWriter.fourcc(*'mp4v'),
                cv2.VideoWriter.fourcc(*'XVID'),
                cv2.VideoWriter.fourcc(*'MJPG'),
                0x7634706d,  # mp4v as integer
                -1  # Let OpenCV choose
            ]
            
            out = None
            for fourcc in fourcc_options:
                try:
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        break
                    out.release()
                except:
                    continue
            
            if out is None or not out.isOpened():
                cap.release()
                raise Exception("Could not initialize video writer for camera extraction")
            
            frame_idx = 0
            assign_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % ASSIGNMENT_STRIDE == 0:
                    if assign_idx < len(assignments_np) and assignments_np[assign_idx] == main_template:
                        out.write(frame)
                    assign_idx += 1
                frame_idx += 1
            
            cap.release()
            out.release()
            
            task.camera_output = output_path
            task.stage = "camera_completed"
            
        except Exception as e:
            task.stage = "failed"
            task.error_message = f"Camera extraction failed: {str(e)}"
            print(f"❌ Camera extraction failed for {task.video_title}: {e}")
        
        return task
    
    def process_player_detection(self, task: ProcessingTask) -> ProcessingTask:
        """Add player detection and pose estimation to video."""
        if self.stop_event.is_set() or task.stage == "failed":
            return task
        
        task.stage = "player"
        
        try:
            # Import player detection modules
            from Players.enhanced_player_detection import EnhancedPlayerDetector
            from Players.court_mapping import SquashCourtMapper
            import cv2
            import numpy as np
            
            # Setup output path
            playlist_output_dir = os.path.join(self.folders['player_annotations'], task.playlist_title)
            os.makedirs(playlist_output_dir, exist_ok=True)
            
            input_path = task.camera_output
            output_filename = f"enhanced_{os.path.basename(input_path)}"
            output_path = os.path.join(playlist_output_dir, output_filename)
            
            # Initialize enhanced detector
            detector = EnhancedPlayerDetector()
            
            # Process video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"Could not open camera output: {input_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate crop dimensions
            crop_margin_percent = 0.10
            crop_margin = int(crop_margin_percent * width)
            cropped_width = width - 2 * crop_margin
            
            # Create court mapper
            court_mapper = SquashCourtMapper(cropped_width, height)
            
            # Setup video writer with robust codec fallback
            fourcc_options = [
                cv2.VideoWriter.fourcc(*'mp4v'),
                cv2.VideoWriter.fourcc(*'XVID'),
                cv2.VideoWriter.fourcc(*'MJPG'),
                0x7634706d,  # mp4v as integer
                -1  # Let OpenCV choose
            ]
            
            out = None
            for fourcc in fourcc_options:
                try:
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        break
                    out.release()
                except:
                    continue
            
            if out is None or not out.isOpened():
                raise Exception("Could not initialize video writer with any codec")
            
            # Process frames
            frame_idx = 0
            cached_player_data = []
            inference_stride = 5
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Crop frame for processing
                cropped_frame = frame[:, crop_margin:width-crop_margin]
                
                # Run enhanced detection every N frames
                if frame_idx % inference_stride == 0:
                    try:
                        annotated_cropped, player_data = detector.process_frame(
                            cropped_frame, show_boundaries=False
                        )
                        cached_player_data = player_data
                    except Exception:
                        # Keep using cached data on error
                        annotated_cropped = cropped_frame.copy()
                else:
                    # Use cached results
                    if cached_player_data:
                        annotated_cropped = cropped_frame.copy()
                        # Draw cached annotations
                        for i, player in enumerate(cached_player_data[:2]):
                            bbox = player['bbox']
                            if bbox:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(annotated_cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_cropped, player['label'], (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Draw pose if available
                                if player['pose'] is not None:
                                    annotated_cropped = detector.draw_stick_figure(annotated_cropped, player['pose'])
                    else:
                        annotated_cropped = cropped_frame.copy()
                
                # Create full frame with annotations
                full_annotated_frame = frame.copy()
                full_annotated_frame[:, crop_margin:width-crop_margin] = annotated_cropped
                
                # Create court overlay
                court_overlay = self._create_court_overlay(court_mapper, cached_player_data)
                
                # Add overlay to frame
                final_frame = self._add_overlay_to_frame(full_annotated_frame, court_overlay)
                
                # Write frame
                out.write(final_frame)
                frame_idx += 1
            
            cap.release()
            out.release()
            
            task.player_output = output_path
            task.stage = "completed"
            
        except Exception as e:
            task.stage = "failed"
            task.error_message = f"Player detection failed: {str(e)}"
            print(f"❌ Player detection failed for {task.video_title}: {e}")
        
        return task
    
    def _create_court_overlay(self, court_mapper, player_data, overlay_size=200):
        """Create court overlay with player positions."""
        overlay = np.ones((overlay_size, overlay_size, 3), dtype=np.uint8) * 255
        
        # Scale factor
        scale_x = overlay_size / court_mapper.court_display_width
        scale_y = overlay_size / court_mapper.court_display_height
        
        # Draw court
        cv2.rectangle(overlay, (2, 2), (overlay_size-3, overlay_size-3), (0, 0, 0), 2)
        
        # Draw court lines
        service_line_y = int(overlay_size * (1 - court_mapper.service_line / court_mapper.court_length))
        cv2.line(overlay, (5, service_line_y), (overlay_size-5, service_line_y), (100, 100, 100), 1)
        
        short_line_y = int(overlay_size * (1 - court_mapper.short_line / court_mapper.court_length))
        cv2.line(overlay, (5, short_line_y), (overlay_size-5, short_line_y), (100, 100, 100), 1)
        
        cv2.line(overlay, (overlay_size//2, 5), (overlay_size//2, overlay_size-5), (100, 100, 100), 1)
        
        # Draw players
        player_colors = [(0, 0, 255), (255, 0, 0)]
        for i, player in enumerate(player_data[:2]):
            if player and player['bbox']:
                court_x, court_y = court_mapper.transform_player_position(player['bbox'])
                if court_x is not None and court_y is not None:
                    overlay_x = int(court_x * scale_x)
                    overlay_y = int(court_y * scale_y)
                    overlay_x = max(5, min(overlay_x, overlay_size-5))
                    overlay_y = max(5, min(overlay_y, overlay_size-5))
                    
                    color = player_colors[i] if i < len(player_colors) else (128, 128, 128)
                    cv2.circle(overlay, (overlay_x, overlay_y), 6, color, -1)
                    cv2.circle(overlay, (overlay_x, overlay_y), 6, (0, 0, 0), 1)
                    cv2.putText(overlay, f'P{i+1}', (overlay_x-8, overlay_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        cv2.putText(overlay, 'Court Map', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return overlay
    
    def _add_overlay_to_frame(self, frame, overlay, overlay_margin=10):
        """Add court overlay to top-right corner."""
        h, w = frame.shape[:2]
        overlay_size = overlay.shape[0]
        
        start_x = w - overlay_size - overlay_margin
        start_y = overlay_margin
        end_x = start_x + overlay_size
        end_y = start_y + overlay_size
        
        if start_x >= 0 and start_y >= 0 and end_x <= w and end_y <= h:
            alpha = 0.9
            frame[start_y:end_y, start_x:end_x] = (
                alpha * overlay + (1 - alpha) * frame[start_y:end_y, start_x:end_x]
            ).astype(np.uint8)
        
        return frame
    
    def process_videos_parallel(self, progress_callback=None):
        """Process all videos in the queue with parallel processing."""
        if self.total_videos == 0:
            print("📭 No videos to process")
            return
        
        print(f"🎬 Starting parallel processing of {self.total_videos} videos...")
        
        completed_tasks = []
        
        # Create progress bar
        with tqdm(total=self.total_videos * 2, desc="Processing pipeline") as pbar:  # *2 for camera + player stages
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Process camera extraction first
                camera_futures = []
                
                # Submit all camera extraction tasks
                while not self.camera_queue.empty():
                    task = self.camera_queue.get()
                    future = executor.submit(self.process_camera_extraction, task)
                    camera_futures.append(future)
                
                # Process camera results and submit player detection
                player_futures = []
                
                for future in as_completed(camera_futures):
                    if self.stop_event.is_set():
                        break
                    
                    task = future.result()
                    pbar.set_postfix_str(f"📹 Camera: {task.video_title[:30]}")
                    pbar.update(1)
                    
                    if task.stage == "camera_completed":
                        # Submit for player detection
                        player_future = executor.submit(self.process_player_detection, task)
                        player_futures.append(player_future)
                    else:
                        self.failed_videos += 1
                        completed_tasks.append(task)
                
                # Process player detection results
                for future in as_completed(player_futures):
                    if self.stop_event.is_set():
                        break
                    
                    task = future.result()
                    pbar.set_postfix_str(f"🤸 Player: {task.video_title[:30]}")
                    pbar.update(1)
                    
                    if task.stage == "completed":
                        self.processed_videos += 1
                        # Copy final output to final_outputs folder
                        self._copy_to_final_outputs(task)
                    else:
                        self.failed_videos += 1
                    
                    completed_tasks.append(task)
                    
                    if progress_callback:
                        progress_callback(task)
        
        # Save processing summary
        self._save_processing_summary(completed_tasks)
        
        print(f"\\n{'='*60}")
        print(f"🎬 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {self.processed_videos}")
        print(f"❌ Failed processing: {self.failed_videos}")
        print(f"📁 Final outputs: {self.folders['final_outputs']}")
    
    def _copy_to_final_outputs(self, task: ProcessingTask):
        """Copy final processed video to final outputs folder."""
        if task.player_output and os.path.exists(task.player_output):
            final_output_dir = os.path.join(self.folders['final_outputs'], task.playlist_title)
            os.makedirs(final_output_dir, exist_ok=True)
            
            final_path = os.path.join(final_output_dir, os.path.basename(task.player_output))
            shutil.copy2(task.player_output, final_path)
    
    def _save_processing_summary(self, completed_tasks: List[ProcessingTask]):
        """Save processing summary to log file."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_processed': self.processed_videos,
            'total_failed': self.failed_videos,
            'tasks': [
                {
                    'video_title': task.video_title,
                    'playlist_title': task.playlist_title,
                    'stage': task.stage,
                    'camera_output': task.camera_output,
                    'player_output': task.player_output,
                    'error_message': task.error_message
                }
                for task in completed_tasks
            ]
        }
        
        log_file = os.path.join(self.folders['logs'], 'processing_summary.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Processing summary saved to: {log_file}")
    
    def stop_processing(self):
        """Stop all processing gracefully."""
        print("🛑 Stopping video processing...")
        self.stop_event.set()