#!/usr/bin/env python3
"""
Clean Squash Video Analysis Pipeline
Uses modular libraries for each component
"""

import os
import sys
import time
import argparse
import cv2
from tqdm import tqdm

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

# Import pipeline libraries
from utils import VideoDownloader, FileManager, DataExporter
from camera_processor import CameraProcessor
from player_detection import PlayerDetector, CourtVisualizer

class SquashPipeline:
    """Clean, modular squash video analysis pipeline"""
    
    def __init__(self, output_dir="Pipeline_Output", write_video=True, max_duration=None):
        self.output_dir = output_dir
        self.write_video = write_video
        self.max_duration = max_duration
        
        # Initialize components
        self.file_manager = FileManager()
        self.camera_processor = CameraProcessor()
        self.player_detector = PlayerDetector()
        self.visualizer = CourtVisualizer()
        self.data_exporter = DataExporter()
        
        # Setup directories
        self.folders = self.file_manager.setup_directories(output_dir)
        
        # Progress tracking
        self.total_videos = 0
        self.processed_videos = 0
        self.failed_videos = 0
        
        print(f"🎬 Squash Pipeline initialized")
        print(f"📁 Output directory: {output_dir}")
        if max_duration:
            print(f"⏱️ Max duration: {max_duration} seconds")
    
    def process_single_video(self, video_url, force_reprocess=False, skip_video_creation=False):
        """Process a single video from URL"""
        print(f"\n{'='*80}")
        print(f"🎯 PROCESSING SINGLE VIDEO")
        print(f"{'='*80}")
        print(f"📹 Video URL: {video_url}")
        
        try:
            # 1. Download video
            print(f"\n📥 Step 1: Downloading video...")
            downloader = VideoDownloader(self.output_dir)
            video_info = downloader.download_video(video_url, "Single Video Analysis")
            
            if not video_info:
                print("❌ Failed to download video")
                return False
            
            print(f"✅ Downloaded: {video_info['title']}")
            self.total_videos = 1
            
            # 2. Extract main camera (check if we can skip this step)
            video_path = video_info['file_path']
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            expected_camera_output = os.path.join(
                self.folders['camera_segments'], 
                "main_camera_segments", 
                f"{base_name}.f399.mp4"
            )
            
            # Check if camera extraction can be skipped
            if not force_reprocess and os.path.exists(expected_camera_output):
                print(f"\n📹 Step 2: Camera extraction already exists, skipping...")
                print(f"   ✅ Using existing camera segments: {os.path.basename(expected_camera_output)}")
                camera_output = expected_camera_output
            else:
                print(f"\n📹 Step 2: Extracting main camera segments...")
                camera_output = self.camera_processor.extract_main_camera_segments(
                    video_path, 
                    self.folders['camera_segments'],
                    max_duration=self.max_duration,
                    force_reprocess=force_reprocess
                )
                
                if not camera_output:
                    print("❌ Camera extraction failed")
                    self.failed_videos += 1
                    return False
            
            # 3. Process player detection and analysis
            print(f"\n🤸 Step 3: Player detection and pose analysis...")
            success = self._process_player_analysis(
                camera_output,
                video_info['title'],
                "Single Video Analysis",
                force_reprocess=force_reprocess,
                skip_video_creation=skip_video_creation
            )
            
            if success:
                self.processed_videos += 1
                print(f"\n✅ Pipeline completed successfully!")
                self._print_summary()
                return True
            else:
                self.failed_videos += 1
                return False
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            self.failed_videos += 1
            return False
    
    def _draw_cached_annotations(self, cropped_frame, cached_player_data, crop_margin):
        """Draw cached player annotations on current frame"""
        try:
            annotated_frame = cropped_frame.copy()
            
            # Debug: Always draw a test rectangle in top-left corner
            cv2.rectangle(annotated_frame, (10, 10), (100, 50), (0, 255, 255), 3)
            cv2.putText(annotated_frame, f"Players: {len(cached_player_data)}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            for i, player in enumerate(cached_player_data):
                bbox = player['bbox']
                confidence = player['confidence']
                
                # The bbox from player detection is already adjusted to the original frame
                # We need to adjust it to the cropped frame coordinates
                adjusted_bbox = [bbox[0] - crop_margin, bbox[1], bbox[2] - crop_margin, bbox[3]]
                
                # Draw bounding box with thicker lines for visibility
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 1, red for player 2
                
                # Ensure bbox coordinates are valid and within frame bounds
                if (len(adjusted_bbox) == 4 and 
                    adjusted_bbox[0] >= 0 and adjusted_bbox[1] >= 0 and
                    adjusted_bbox[2] < cropped_frame.shape[1] and adjusted_bbox[3] < cropped_frame.shape[0]):
                    
                    cv2.rectangle(annotated_frame, (adjusted_bbox[0], adjusted_bbox[1]), 
                                (adjusted_bbox[2], adjusted_bbox[3]), color, 4)
                    
                    # Draw label
                    label = f"{player['label']} ({confidence:.2f})"
                    cv2.putText(annotated_frame, label, (adjusted_bbox[0], adjusted_bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            return annotated_frame
        except Exception as e:
            print(f"❌ Cached annotation error: {e}")
            return cropped_frame
    
    def _process_player_analysis(self, input_path, video_title, playlist_title, force_reprocess=False, skip_video_creation=False):
        """Process player detection and create annotated video with data export"""
        try:
            # Setup output paths
            playlist_output_dir = os.path.join(
                self.folders['player_annotations'], 
                self.file_manager.sanitize_filename(playlist_title)
            )
            os.makedirs(playlist_output_dir, exist_ok=True)
            
            output_filename = f"enhanced_{self.file_manager.sanitize_filename(os.path.basename(input_path))}"
            output_path = os.path.join(playlist_output_dir, output_filename)

            # If force_reprocess, remove existing output video and CSV
            if force_reprocess:
                if os.path.exists(output_path):
                    os.remove(output_path)
                csv_path = os.path.join(playlist_output_dir, f"{video_title}_player_data.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)

            # Check if video output already exists
            video_already_exists = self.file_manager.video_exists(output_path)
            
            # Determine skipping behavior based on flags
            skip_video_creation_flag = False
            if skip_video_creation:
                # User explicitly wants to skip video creation
                skip_video_creation_flag = True
                print(f"   ⚠️ Skipping video creation (--skip-video flag): {output_filename}")
            elif video_already_exists and self.write_video and not force_reprocess:
                # Video exists and user hasn't forced reprocessing
                skip_video_creation_flag = True
                print(f"   ⚠️ Video exists, skipping video creation but regenerating CSV: {output_filename}")
            
            # Always process for CSV data (even if video exists or is skipped)
            
            # Player detector is already initialized in __init__
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"   ❌ Cannot open video: {input_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"   📊 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
            print(f"   📊 Output will be: {width}x{height} (full resolution)")
            
            # Setup video writer (only if we need to create video)
            out = None
            if self.write_video and not skip_video_creation_flag:
                out = self.file_manager.create_video_writer(output_path, fps, width, height)
                if not out:
                    print(f"   ⚠️ Video writing failed, continuing with CSV-only mode")
            
            # Processing parameters
            frame_idx = 0
            inference_stride = 10  # Process every 10th frame for efficiency (increased from 5)
            cached_player_data = []
            csv_data = []
            
            # Setup progress bar - process all frames, but stop at max_duration if specified
            progress_bar = tqdm(total=total_frames, desc="   🎬 Processing frames")
            
            # Main processing loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check time limit
                if self.max_duration:
                    current_time = frame_idx / fps if fps > 0 else frame_idx / 25
                    if current_time > self.max_duration:
                        print(f"   ⏱️ Stopping at {self.max_duration} seconds")
                        break
                
                # Process frame for player detection
                if frame_idx % inference_stride == 0:
                    # Run full detection
                    annotated_frame, player_data = self.player_detector.process_frame(frame, frame_idx=frame_idx)
                    cached_player_data = player_data
                else:
                    # On skipped frames, do NOT draw any player annotations
                    annotated_frame = frame.copy()
                
                # Court mapping is already done in player_detector.process_frame
                
                # Create court overlay
                court_overlay = self.visualizer.create_court_overlay(cached_player_data)
                final_frame = self.visualizer.add_overlay_to_frame(annotated_frame, court_overlay)
                
                # Collect data for CSV
                timestamp = frame_idx / fps if fps > 0 else frame_idx
                frame_data = self.data_exporter.collect_frame_data(
                    frame_idx, timestamp, cached_player_data, video_title
                )
                csv_data.append(frame_data)
                
                # Write frame if video output enabled
                if out:
                    out.write(final_frame)
                
                frame_idx += 1
                progress_bar.update(1)
            
            # Cleanup
            progress_bar.close()
            cap.release()
            if out:
                out.release()
            
            # Save CSV data
            csv_path = self.data_exporter.save_to_csv(csv_data, playlist_output_dir, video_title)
            
            # Copy to final outputs (only if video was created or already exists)
            if self.write_video and os.path.exists(output_path):
                final_output_dir = os.path.join(self.folders['final_outputs'], playlist_title)
                os.makedirs(final_output_dir, exist_ok=True)
                final_path = os.path.join(final_output_dir, os.path.basename(output_path))

                # If force_reprocess, remove existing final output
                if force_reprocess and os.path.exists(final_path):
                    os.remove(final_path)

                import shutil
                shutil.copy2(output_path, final_path)
                print(f"   📁 Copied to: {os.path.basename(final_path)}")
            
            if skip_video_creation_flag:
                if skip_video_creation:
                    print(f"   ✅ CSV generated: {len(csv_data)} frames processed (video creation skipped)")
                else:
                    print(f"   ✅ CSV regenerated: {len(csv_data)} frames processed (video already exists)")
            else:
                print(f"   ✅ Analysis complete: {len(csv_data)} frames processed")
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Player analysis failed: {e}")
            return False
    
    def _print_summary(self):
        """Print pipeline summary"""
        print(f"\n{'='*80}")
        print(f"🎉 PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Videos processed: {self.processed_videos}/{self.total_videos}")
        print(f"❌ Failed videos: {self.failed_videos}")
        
        if self.total_videos > 0:
            success_rate = (self.processed_videos / self.total_videos) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        print(f"\n📁 Output locations:")
        print(f"   🎬 Final videos: {self.folders['final_outputs']}")
        print(f"   📊 CSV data: {self.folders['player_annotations']}")
        print(f"   📹 Camera segments: {self.folders['camera_segments']}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Clean Squash Video Analysis Pipeline")
    parser.add_argument("video_url", help="YouTube video URL")
    parser.add_argument("--output-dir", default="Pipeline_Output", help="Output directory")
    parser.add_argument("--csv-only", action="store_true", 
                       help="Extract data to CSV without writing video files")
    parser.add_argument("--max-duration", type=float, default=None,
                       help="Maximum duration to process in seconds")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing even if output video exists")
    parser.add_argument("--skip-video", action="store_true", 
                       help="Skip video creation, only generate CSV (faster)")
    
    args = parser.parse_args()
    
    # Create pipeline
    write_video = not args.csv_only
    pipeline = SquashPipeline(
        output_dir=args.output_dir,
        write_video=write_video,
        max_duration=args.max_duration
    )
    
    if args.csv_only:
        print("🔥 CSV-only mode: Video writing disabled for faster processing")
    
    # Process the video
    success = pipeline.process_single_video(
        args.video_url,
        force_reprocess=args.force_reprocess,
        skip_video_creation=args.skip_video
    )
    
    if not success:
        print("\n❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()