import cv2
import numpy as np
import os
from tqdm import tqdm
from enhanced_player_detection import EnhancedPlayerDetector
from court_mapping import SquashCourtMapper

class EnhancedVideoOverlay:
    """
    Enhanced video processing with pose detection and court boundary constraints.
    """
    
    def __init__(self, yolo_model_path='yolov8n.pt', crop_margin_percent=0.10):
        self.crop_margin_percent = crop_margin_percent
        
        # Initialize enhanced detector
        self.detector = EnhancedPlayerDetector(yolo_model_path)
        
        # Overlay settings
        self.overlay_size = 200
        self.overlay_margin = 10
        
    def create_court_overlay(self, court_mapper, player_data):
        """Create court overlay with player positions."""
        overlay = np.ones((self.overlay_size, self.overlay_size, 3), dtype=np.uint8) * 255
        
        # Scale factor for court to overlay
        scale_x = self.overlay_size / court_mapper.court_display_width
        scale_y = self.overlay_size / court_mapper.court_display_height
        
        # Draw court boundaries
        cv2.rectangle(overlay, (2, 2), (self.overlay_size-3, self.overlay_size-3), (0, 0, 0), 2)
        
        # Draw court lines
        service_line_y = int(self.overlay_size * (1 - court_mapper.service_line / court_mapper.court_length))
        cv2.line(overlay, (5, service_line_y), (self.overlay_size-5, service_line_y), (100, 100, 100), 1)
        
        short_line_y = int(self.overlay_size * (1 - court_mapper.short_line / court_mapper.court_length))
        cv2.line(overlay, (5, short_line_y), (self.overlay_size-5, short_line_y), (100, 100, 100), 1)
        
        cv2.line(overlay, (self.overlay_size//2, 5), (self.overlay_size//2, self.overlay_size-5), (100, 100, 100), 1)
        
        # Draw players
        player_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        
        for i, player in enumerate(player_data):
            if i >= len(player_colors):
                break
                
            bbox = player['bbox']
            if bbox is not None:
                # Transform bbox to court coordinates
                court_x, court_y = court_mapper.transform_player_position(bbox)
                
                if court_x is not None and court_y is not None:
                    # Scale position to overlay
                    overlay_x = int(court_x * scale_x)
                    overlay_y = int(court_y * scale_y)
                    
                    # Ensure within bounds
                    overlay_x = max(5, min(overlay_x, self.overlay_size-5))
                    overlay_y = max(5, min(overlay_y, self.overlay_size-5))
                    
                    # Draw player dot
                    color = player_colors[i]
                    cv2.circle(overlay, (overlay_x, overlay_y), 6, color, -1)
                    cv2.circle(overlay, (overlay_x, overlay_y), 6, (0, 0, 0), 1)
                    
                    # Add player label
                    cv2.putText(overlay, f'P{i+1}', (overlay_x-8, overlay_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(overlay, 'Court Map', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return overlay
    
    def add_overlay_to_frame(self, frame, overlay):
        """Add court overlay to top-right corner."""
        h, w = frame.shape[:2]
        
        start_x = w - self.overlay_size - self.overlay_margin
        start_y = self.overlay_margin
        end_x = start_x + self.overlay_size
        end_y = start_y + self.overlay_size
        
        if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
            return frame
        
        alpha = 0.9
        frame[start_y:end_y, start_x:end_x] = (
            alpha * overlay + (1 - alpha) * frame[start_y:end_y, start_x:end_x]
        ).astype(np.uint8)
        
        return frame
    
    def process_video(self, input_path, output_path, show_progress=True, 
                     show_boundaries=False, inference_stride=5):
        """Process video with enhanced detection and pose estimation."""
        
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
        
        # Setup video writer
        try:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        except AttributeError:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            except AttributeError:
                fourcc = 0x7634706d
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not initialize video writer for {output_path}")
            cap.release()
            return False
        
        # Initialize tracking
        cached_player_data = []
        
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
                
                # Crop frame for processing
                cropped_frame = frame[:, crop_margin:width-crop_margin]
                
                # Run enhanced detection every N frames
                if frame_idx % inference_stride == 0:
                    try:
                        annotated_cropped, player_data = self.detector.process_frame(
                            cropped_frame, show_boundaries=show_boundaries
                        )
                        cached_player_data = player_data
                    except Exception as e:
                        print(f"Warning: Error in enhanced detection for frame {frame_idx}: {e}")
                        # Keep using cached data
                else:
                    # Use cached results and apply to current frame
                    if cached_player_data:
                        annotated_cropped = cropped_frame.copy()
                        # Draw cached annotations
                        for i, player in enumerate(cached_player_data[:2]):  # Limit to 2 players
                            bbox = player['bbox']
                            if bbox:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(annotated_cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_cropped, player['label'], (x1, y1 - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                
                                # Draw cached pose if available
                                if player['pose'] is not None:
                                    annotated_cropped = self.detector.draw_stick_figure(annotated_cropped, player['pose'])
                    else:
                        annotated_cropped = cropped_frame.copy()
                
                # Create full frame with cropped annotations
                full_annotated_frame = frame.copy()
                full_annotated_frame[:, crop_margin:width-crop_margin] = annotated_cropped
                
                # Create court overlay
                court_overlay = self.create_court_overlay(court_mapper, cached_player_data)
                
                # Add overlay to frame
                final_frame = self.add_overlay_to_frame(full_annotated_frame, court_overlay)
                
                # Write frame
                out.write(final_frame)
                
                if progress_iter:
                    progress_iter.update(1)
                
                frame_idx += 1
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return False
        finally:
            cap.release()
            out.release()
            if progress_iter:
                progress_iter.close()
        
        return True


def process_all_videos_enhanced():
    """Process all videos with enhanced detection."""
    input_dir = "../Camera/camera_outputs/main_camera_segments"
    output_dir = "enhanced_court_overlay_videos"
    
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} not found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process with enhanced detection")
    
    processor = EnhancedVideoOverlay()
    
    successful = 0
    failed = 0
    
    for video_file in tqdm(video_files, desc="Processing videos with pose detection"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, f"enhanced_{video_file}")
        
        # Remove existing file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Removed existing file: {video_file}")
        
        print(f"\nProcessing with pose detection: {video_file}")
        
        try:
            success = processor.process_video(
                input_path=input_path,
                output_path=output_path,
                show_progress=False,
                show_boundaries=False,  # Set to True for debugging
                inference_stride=3  # Run detection every 3rd frame
            )
            
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
    print(f"🎉 ENHANCED PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total videos: {len(video_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤸 Features: Pose detection + Court boundaries + Stick figures")
    
    if failed > 0:
        print(f"\n⚠️  Check error messages for failed videos.")
    else:
        print(f"\n🎯 All videos processed with enhanced detection!")


if __name__ == "__main__":
    process_all_videos_enhanced()