import cv2
import numpy as np
from player_detection_lib import detect_players, assign_player_labels, load_yolo_model
from court_mapping import SquashCourtMapper
import os
from tqdm import tqdm

class VideoCourtOverlay:
    """
    Creates videos with court map overlay showing player positions in real-time.
    The court map appears in the top-right corner of the video.
    """
    
    def __init__(self, model_path='yolov8n.pt', crop_margin_percent=0.10):
        self.model_path = model_path
        self.crop_margin_percent = crop_margin_percent
        self.model = load_yolo_model(model_path)
        
        # Overlay settings
        self.overlay_size = 200  # Size of the court overlay
        self.overlay_margin = 10  # Margin from edges
        
    def create_court_overlay(self, court_mapper, player_positions):
        """Create a court overlay image with player positions."""
        # Create white background
        overlay = np.ones((self.overlay_size, self.overlay_size, 3), dtype=np.uint8) * 255
        
        # Scale factor for court to overlay
        scale_x = self.overlay_size / court_mapper.court_display_width
        scale_y = self.overlay_size / court_mapper.court_display_height
        
        # Draw court boundaries
        cv2.rectangle(overlay, (2, 2), (self.overlay_size-3, self.overlay_size-3), 
                     (0, 0, 0), 2)
        
        # Draw court lines
        # Service line
        service_line_y = int(self.overlay_size * (1 - court_mapper.service_line / court_mapper.court_length))
        cv2.line(overlay, (5, service_line_y), (self.overlay_size-5, service_line_y), 
                (100, 100, 100), 1)
        
        # Short line
        short_line_y = int(self.overlay_size * (1 - court_mapper.short_line / court_mapper.court_length))
        cv2.line(overlay, (5, short_line_y), (self.overlay_size-5, short_line_y), 
                (100, 100, 100), 1)
        
        # Center line
        cv2.line(overlay, (self.overlay_size//2, 5), (self.overlay_size//2, self.overlay_size-5), 
                (100, 100, 100), 1)
        
        # Draw players
        player_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]  # Red, Blue, Green, Yellow in BGR
        for i, (x, y) in enumerate(player_positions):
            if x is not None and y is not None and i < len(player_colors):
                # Scale position to overlay size
                overlay_x = int(x * scale_x)
                overlay_y = int(y * scale_y)
                
                # Ensure within bounds
                overlay_x = max(5, min(overlay_x, self.overlay_size-5))
                overlay_y = max(5, min(overlay_y, self.overlay_size-5))
                
                # Draw player dot
                color = player_colors[i] if i < len(player_colors) else (128, 128, 128)  # Default gray
                cv2.circle(overlay, (overlay_x, overlay_y), 6, color, -1)
                cv2.circle(overlay, (overlay_x, overlay_y), 6, (0, 0, 0), 1)
                
                # Add player label
                cv2.putText(overlay, f'P{i+1}', (overlay_x-8, overlay_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(overlay, 'Court Map', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return overlay
    
    def add_overlay_to_frame(self, frame, overlay):
        """Add the court overlay to the top-right corner of the frame."""
        h, w = frame.shape[:2]
        
        # Position for top-right corner
        start_x = w - self.overlay_size - self.overlay_margin
        start_y = self.overlay_margin
        end_x = start_x + self.overlay_size
        end_y = start_y + self.overlay_size
        
        # Ensure overlay fits within frame
        if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
            return frame  # Skip overlay if it doesn't fit
        
        # Add semi-transparent background
        alpha = 0.9
        frame[start_y:end_y, start_x:end_x] = (
            alpha * overlay + (1 - alpha) * frame[start_y:end_y, start_x:end_x]
        ).astype(np.uint8)
        
        return frame
    
    def process_video(self, input_path, output_path, show_progress=True):
        """Process a video file and add court overlay."""
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate crop dimensions
        crop_margin = int(self.crop_margin_percent * width)
        cropped_width = width - 2 * crop_margin
        
        # Create court mapper
        court_mapper = SquashCourtMapper(cropped_width, height)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup video writer with better compatibility
        try:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        except AttributeError:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except AttributeError:
                fourcc = 0x7634706d  # 'mp4v' as integer
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if video writer was successfully initialized
        if not out.isOpened():
            print(f"Error: Could not initialize video writer for {output_path}")
            cap.release()
            return False
        
        # Initialize player tracking
        prev_centroids = None
        
        # Initialize cached variables
        cached_court_positions = [(None, None), (None, None)]
        cached_player_boxes = []
        cached_labels = []
        
        # Progress tracking
        progress_iter = None
        if show_progress:
            progress_iter = tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_path)}")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Crop frame for player detection
                cropped_frame = frame[:, crop_margin:width-crop_margin]
                
                # Detect players (run every 3rd frame for performance)
                if frame_idx % 3 == 0:
                    try:
                        player_boxes, centroids = detect_players(self.model, cropped_frame)
                        labels, prev_centroids = assign_player_labels(centroids, prev_centroids)
                        
                        # Transform to court coordinates
                        court_positions = []
                        for i, bbox in enumerate(player_boxes):
                            if len(bbox) == 4:
                                try:
                                    court_x, court_y = court_mapper.transform_player_position(bbox)
                                    court_positions.append((court_x, court_y))
                                except Exception as e:
                                    court_positions.append((None, None))
                            else:
                                court_positions.append((None, None))
                        
                        # Ensure we have 2 positions for consistency
                        while len(court_positions) < 2:
                            court_positions.append((None, None))
                        
                        # Store for next frames (limit to first 2 players for consistency)
                        cached_court_positions = court_positions[:2]
                        cached_player_boxes = player_boxes[:2]  # Limit to 2 players
                        cached_labels = labels[:2]  # Limit to 2 players
                        
                    except Exception as e:
                        print(f"Warning: Error in player detection for frame {frame_idx}: {e}")
                        # Keep using previous cached values
                else:
                    # Use cached results
                    court_positions = cached_court_positions
                    player_boxes = cached_player_boxes
                    labels = cached_labels
                
                # Draw player annotations on original frame
                annotated_frame = frame.copy()
                
                if len(player_boxes) > 0 and len(labels) > 0:
                    # Only draw annotations for detected players (max 2)
                    min_length = min(len(player_boxes), len(labels), 2)
                    
                    for i in range(min_length):
                        try:
                            bbox = player_boxes[i]
                            label = labels[i]
                            
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                # Adjust coordinates for full frame (add crop margin back)
                                x1 += crop_margin
                                x2 += crop_margin
                                
                                label_str = label if label else f"Player {i+1}"
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, label_str, (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                        except Exception as e:
                            print(f"Frame {frame_idx}: Error drawing player {i}: {e}")
                            continue
                
                # Create court overlay
                court_overlay = self.create_court_overlay(court_mapper, court_positions)
                
                # Add overlay to frame
                final_frame = self.add_overlay_to_frame(annotated_frame, court_overlay)
                
                # Write frame
                out.write(final_frame)
                
                if progress_iter:
                    progress_iter.update(1)
                
                frame_idx += 1
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return False
        
        # Cleanup
        cap.release()
        out.release()
        if progress_iter:
            progress_iter.close()
        
        return True


def process_all_videos():
    """Process all videos in the main camera segments directory."""
    input_dir = "../Camera/camera_outputs/main_camera_segments"
    output_dir = "court_overlay_videos"
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create processor
    processor = VideoCourtOverlay()
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"court_overlay_{video_file}")
        
        # Remove existing file if it exists (reprocess all videos)
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Removed existing file: {video_file}")
        
        print(f"\nProcessing: {video_file}")
        
        try:
            success = processor.process_video(input_path, output_path, show_progress=False)
            if success:
                successful += 1
                print(f"✓ Successfully processed: {video_file}")
            else:
                failed += 1
                print(f"✗ Failed to process: {video_file}")
        except Exception as e:
            failed += 1
            print(f"✗ Error processing {video_file}: {str(e)}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    process_all_videos()