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
import subprocess

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

# Import pipeline libraries
from utils import VideoDownloader, FileManager, DataExporter, create_frame_difference_video, create_motion_heatmap_video, create_multi_duration_heatmap_grid, create_frame_diff_comparison_grid, detect_players_and_ball_from_motion, create_frame_diff_and_detect_players
from camera_processor import CameraProcessor
from player_detection import PlayerDetector, CourtVisualizer
from audio_events import detect_hits_and_walls, annotate_video_with_events

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
            
            # 3. Create frame difference video for motion analysis
            print(f"\n🎬 Step 3: Creating frame difference video...")
            base_name = os.path.splitext(os.path.basename(camera_output))[0]
            diff_video_path = os.path.join(
                self.folders['camera_segments'],
                "frame_difference",
                f"{base_name}_frame_diff.mp4"
            )
            os.makedirs(os.path.dirname(diff_video_path), exist_ok=True)
            
            # Only create if it doesn't exist or force reprocess
            if not os.path.exists(diff_video_path) or force_reprocess:
                diff_result = create_frame_difference_video(camera_output, diff_video_path, max_duration=self.max_duration)
                if diff_result:
                    print(f"   ✅ Frame difference video created: {os.path.basename(diff_video_path)}")
                else:
                    print(f"   ⚠️ Frame difference video creation failed")
            else:
                print(f"   ✅ Frame difference video already exists: {os.path.basename(diff_video_path)}")
            
            # 4. Create motion heatmap video
            print(f"\n🔥 Step 4: Creating motion heatmap video...")
            heatmap_video_path = os.path.join(
                self.folders['camera_segments'],
                "motion_heatmap", 
                f"{base_name}_motion_heatmap.mp4"
            )
            os.makedirs(os.path.dirname(heatmap_video_path), exist_ok=True)
            
            # Only create if it doesn't exist or force reprocess
            if not os.path.exists(heatmap_video_path) or force_reprocess:
                heatmap_result = create_motion_heatmap_video(camera_output, heatmap_video_path, max_duration=self.max_duration)
                if heatmap_result:
                    print(f"   ✅ Motion heatmap video created: {os.path.basename(heatmap_video_path)}")
                else:
                    print(f"   ⚠️ Motion heatmap video creation failed")
            else:
                print(f"   ✅ Motion heatmap video already exists: {os.path.basename(heatmap_video_path)}")

            # 5. Process player detection and analysis
            print(f"\n🤸 Step 5: Player detection and pose analysis...")
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
            cached_candidates = []
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
                    annotated_frame, player_data, all_candidates = self.player_detector.process_frame(frame, frame_idx=frame_idx)
                    cached_player_data = player_data
                    cached_candidates = all_candidates
                else:
                    # On skipped frames, do NOT draw any player annotations
                    annotated_frame = frame.copy()
                    cached_candidates = []
                
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
            
            # After player analysis, run audio event detection
            print(f"\n🔊 Step 4: Detecting ball hits and wall events in audio...")
            events = detect_hits_and_walls(output_path)
            n_hits = sum(1 for t, e in events if e == 'hit')
            n_walls = sum(1 for t, e in events if e == 'wall')
            print(f"   🎯 Detected {n_hits} hits and {n_walls} wall events in audio.")
            # Annotate video with detected events
            print(f"   📝 Annotating video with audio events...")
            annotated_video_path = annotate_video_with_events(output_path, events)
            print(f"   ✅ Audio event-annotated video: {annotated_video_path}")
            
            # After writing the processed video (output_path), merge audio if possible
            # Find the original downloaded video with audio
            orig_video_path = self.file_manager.get_downloaded_video_path(video_title)
            if orig_video_path and os.path.exists(orig_video_path):
                # Check if original video has audio
                has_audio = self.file_manager.video_has_audio(orig_video_path)
                if has_audio:
                    self._merge_audio_into_video(output_path, orig_video_path, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Player analysis failed: {e}")
            return False
    
    def _merge_audio_into_video(self, video_path, audio_source_path, output_path=None):
        """Merge audio from audio_source_path into video_path, write to output_path (or overwrite video_path)."""
        if output_path is None:
            output_path = video_path
        # Use ffmpeg to merge audio
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', audio_source_path,
            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"   🔊 Merged audio from {audio_source_path} into {output_path}")
            return output_path
        except Exception as e:
            print(f"   ❌ Failed to merge audio: {e}")
            return video_path
    
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
    parser = argparse.ArgumentParser(description="Squash Video Analysis Pipeline")
    parser.add_argument("video_url", type=str, nargs="?", help="YouTube video URL or local video file")
    parser.add_argument("--max-duration", type=int, default=None, help="Max duration (seconds) to process")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of all outputs")
    parser.add_argument("--audio-events-only", action="store_true", help="Only run audio event detection/annotation on a video file")
    parser.add_argument("--audio-video-path", type=str, default=None, help="Path to video file for audio event detection (if using --audio-events-only)")
    parser.add_argument("--frame-diff-only", action="store_true", help="Only create frame difference and motion heatmap videos")
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file for frame difference processing")
    parser.add_argument("--heatmap-grid", action="store_true", help="Create 3x3 grid of motion heatmaps with different durations")
    parser.add_argument("--diff-grid", action="store_true", help="Create 3x3 grid of frame differences with different thresholds")
    parser.add_argument("--motion-detect", action="store_true", help="Detect players and ball using frame difference analysis")
    parser.add_argument("--motion-threshold", type=int, default=10, help="Threshold for motion detection (default: 10)")
    parser.add_argument("--framediff-detect", action="store_true", help="Run YOLO human detection on frame difference video")
    parser.add_argument("--diff-threshold", type=int, default=10, help="Threshold for frame difference (default: 10)")
    args = parser.parse_args()

    if args.audio_events_only:
        # Only run audio event detection/annotation
        video_path = args.audio_video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --audio-video-path or as the positional argument.")
            return
        print(f"\n🔊 Running audio event detection/annotation on: {video_path}")
        events = detect_hits_and_walls(video_path)
        n_hits = sum(1 for t, e in events if e == 'hit')
        n_walls = sum(1 for t, e in events if e == 'wall')
        print(f"   🎯 Detected {n_hits} hits and {n_walls} wall events in audio.")
        print(f"   📝 Annotating video with audio events...")
        annotated_video_path = annotate_video_with_events(video_path, events)
        print(f"   ✅ Audio event-annotated video: {annotated_video_path}")
        return

    if args.frame_diff_only:
        # Only create frame difference and motion heatmap videos
        video_path = args.video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --video-path or as the positional argument.")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🎬 Creating frame difference and motion heatmap videos for: {video_path}")
        
        # Create output directories
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        
        # Create frame difference video
        diff_video_path = os.path.join(output_dir, f"{base_name}_frame_diff.mp4")
        print(f"\n🎬 Creating frame difference video...")
        diff_result = create_frame_difference_video(video_path, diff_video_path, max_duration=args.max_duration)
        
        if diff_result:
            print(f"   ✅ Frame difference video created: {diff_video_path}")
        else:
            print(f"   ❌ Frame difference video creation failed")
        
        # Create motion heatmap video
        heatmap_video_path = os.path.join(output_dir, f"{base_name}_motion_heatmap.mp4")
        print(f"\n🔥 Creating motion heatmap video...")
        heatmap_result = create_motion_heatmap_video(video_path, heatmap_video_path, max_duration=args.max_duration)
        
        if heatmap_result:
            print(f"   ✅ Motion heatmap video created: {heatmap_video_path}")
        else:
            print(f"   ❌ Motion heatmap video creation failed")
        
        print(f"\n✅ Frame difference processing completed!")
        return

    if args.heatmap_grid:
        # Create 3x3 grid of motion heatmaps with different durations
        video_path = args.video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --video-path or as the positional argument.")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🔥 Creating motion heatmap grid for: {video_path}")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        grid_video_path = os.path.join(output_dir, f"{base_name}_heatmap_grid.mp4")
        
        result = create_multi_duration_heatmap_grid(video_path, grid_video_path, max_duration=args.max_duration)
        
        if result:
            print(f"   ✅ Heatmap grid video created: {grid_video_path}")
        else:
            print(f"   ❌ Heatmap grid creation failed")
        
        return

    if args.diff_grid:
        # Create 3x3 grid of frame differences with different thresholds
        video_path = args.video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --video-path or as the positional argument.")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🎬 Creating frame difference grid for: {video_path}")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        grid_video_path = os.path.join(output_dir, f"{base_name}_diff_grid.mp4")
        
        result = create_frame_diff_comparison_grid(video_path, grid_video_path, max_duration=args.max_duration)
        
        if result:
            print(f"   ✅ Diff grid video created: {grid_video_path}")
        else:
            print(f"   ❌ Diff grid creation failed")
        
        return

    if args.motion_detect:
        # Detect players and ball using frame difference analysis
        video_path = args.video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --video-path or as the positional argument.")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🎯 Detecting players and ball from motion in: {video_path}")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        detection_video_path = os.path.join(output_dir, f"{base_name}_motion_detection.mp4")
        
        results = detect_players_and_ball_from_motion(
            video_path, 
            detection_video_path, 
            diff_threshold=args.motion_threshold,
            max_duration=args.max_duration
        )
        
        if results:
            print(f"   ✅ Motion detection video created: {detection_video_path}")
            
            # Print some statistics
            total_frames = results['total_frames']
            player_frames = sum(1 for frame_players in results['player_detections'] if frame_players)
            ball_frames = sum(1 for frame_ball in results['ball_detections'] if frame_ball)
            
            print(f"   📊 Statistics:")
            print(f"      Total frames: {total_frames}")
            print(f"      Frames with players: {player_frames} ({player_frames/total_frames*100:.1f}%)")
            print(f"      Frames with ball: {ball_frames} ({ball_frames/total_frames*100:.1f}%)")
            
        else:
            print(f"   ❌ Motion detection failed")
        
        return

    if args.framediff_detect:
        # Run YOLO human detection on frame difference video
        video_path = args.video_path or args.video_url
        if not video_path:
            print("❌ Please provide a video file with --video-path or as the positional argument.")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        print(f"\n🎯 Running YOLO detection on frame difference video: {video_path}")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        detection_video_path = os.path.join(output_dir, f"{base_name}_framediff_detection.mp4")
        
        results = create_frame_diff_and_detect_players(
            video_path, 
            detection_video_path, 
            diff_threshold=args.diff_threshold,
            max_duration=args.max_duration
        )
        
        if results:
            print(f"   ✅ Frame difference detection video created: {detection_video_path}")
            print(f"   📊 Found players in {results['total_frames']} frames")
            
        else:
            print(f"   ❌ Frame difference detection failed")
        
        return

    # Create pipeline
    write_video = True
    pipeline = SquashPipeline(
        output_dir="Pipeline_Output",
        write_video=write_video,
        max_duration=args.max_duration
    )
    
    # Process the video
    success = pipeline.process_single_video(
        args.video_url,
        force_reprocess=args.force_reprocess
    )
    
    if not success:
        print("\n❌ Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()