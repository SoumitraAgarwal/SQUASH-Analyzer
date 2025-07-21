"""
Utilities Library
Video downloading, file management, and data export utilities
"""

import os
import re
import cv2
import numpy as np
import yt_dlp
import pandas as pd
from datetime import datetime
import soundfile as sf
import librosa

class VideoDownloader:
    """Simple video downloader for single videos"""
    
    def __init__(self, output_dir="Pipeline_Output"):
        self.output_dir = output_dir
        self.downloads_dir = os.path.join(output_dir, "01_Downloads")
        os.makedirs(self.downloads_dir, exist_ok=True)
    
    def download_video(self, video_url, folder_name="Single Video"):
        """Download a single video"""
        try:
            # Create folder for this video
            folder_path = os.path.join(self.downloads_dir, self._sanitize_filename(folder_name))
            os.makedirs(folder_path, exist_ok=True)
            
            # Get video info first
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', 'Unknown') if info else 'Unknown'
                duration = info.get('duration', 0) if info else 0
                
                print(f"📹 Downloading: {title}")
                
                # Step 1: Download high quality video (video-only)
                video_ydl_opts = {
                    'outtmpl': os.path.join(folder_path, '%(title)s_video.%(ext)s'),
                    'format': (
                        # Try specific high quality formats first (video-only)
                        '299/298/136/'  # 1080p60, 720p60, 720p
                        'best[height>=1080][fps>=60]/best[height>=1080]/'  # Generic 1080p60, 1080p
                        'best[height>=720][fps>=60]/best[height>=720]/'    # Generic 720p60, 720p
                        'bestvideo'  # Best video quality
                    ),
                    'quiet': True,
                    'no_warnings': True,
                    'playliststart': 1,
                    'playlistend': 1,
                }
                
                # Step 2: Download audio separately
                audio_ydl_opts = {
                    'outtmpl': os.path.join(folder_path, '%(title)s_audio.%(ext)s'),
                    'format': (
                        # Try to get best audio quality
                        '140/251/250/249/'  # AAC and Opus audio formats
                        'bestaudio'  # Best audio quality
                    ),
                    'quiet': True,
                    'no_warnings': True,
                    'playliststart': 1,
                    'playlistend': 1,
                }
                
                # Download video
                print("🎬 Downloading high-quality video...")
                with yt_dlp.YoutubeDL(video_ydl_opts) as video_ydl:
                    video_ydl.download([video_url])
                
                # Download audio  
                print("🔊 Downloading audio...")
                with yt_dlp.YoutubeDL(audio_ydl_opts) as audio_ydl:
                    audio_ydl.download([video_url])
                
                # Step 3: Combine video and audio using Python (without ffmpeg)
                print("🔗 Combining video and audio...")
                final_path = self._combine_video_audio(folder_path, title)
                
                if final_path and os.path.exists(final_path):
                    # Trim video to actual duration to remove trailing frames
                    trimmed_file = self._trim_video_to_duration(final_path, duration)
                    
                    return {
                        'title': title,
                        'file_path': trimmed_file,
                        'duration': duration
                    }
                
                print("❌ Failed to combine video and audio")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def _combine_video_audio(self, folder_path, title):
        """Combine video and audio files using Python libraries"""
        import glob
        
        # Find video and audio files
        video_files = glob.glob(os.path.join(folder_path, f"*_video.*"))
        audio_files = glob.glob(os.path.join(folder_path, f"*_audio.*"))
        
        if not video_files or not audio_files:
            print("❌ Could not find both video and audio files")
            return None
        
        video_path = video_files[0]
        audio_path = audio_files[0]
        
        # Create final output path
        safe_title = self._sanitize_filename(title)
        final_path = os.path.join(folder_path, f"{safe_title}.mp4")
        
        try:
            # Try using moviepy if available
            try:
                from moviepy.editor import VideoFileClip, AudioFileClip
                
                print("🎬 Using moviepy to combine video and audio...")
                video_clip = VideoFileClip(video_path)
                audio_clip = AudioFileClip(audio_path)
                
                # Set audio to video
                final_clip = video_clip.set_audio(audio_clip)
                
                # Write final video
                final_clip.write_videofile(final_path, codec='libx264', audio_codec='aac')
                
                # Clean up
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                
                print(f"✅ Combined video saved: {os.path.basename(final_path)}")
                
                # Clean up separate files
                os.remove(video_path)
                os.remove(audio_path)
                
                return final_path
                
            except ImportError:
                # Fallback: Just use the video file and extract audio separately for analysis
                print("⚠️ moviepy not available, using video-only file")
                print("🔊 Audio will be extracted separately for analysis")
                
                # Just copy the video file to the final location
                import shutil
                shutil.move(video_path, final_path)
                
                # Keep the audio file for later processing
                audio_final_path = os.path.join(folder_path, f"{safe_title}_audio.{audio_path.split('.')[-1]}")
                shutil.move(audio_path, audio_final_path)
                
                return final_path
                
        except Exception as e:
            print(f"❌ Error combining video and audio: {e}")
            # Fallback to video-only
            import shutil
            shutil.move(video_path, final_path)
            return final_path
    
    def _sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        # Remove invalid characters AND Unicode pipe characters
        filename = re.sub(r'[<>:"/\\|?*｜⧸]', '', filename)
        # Remove emoji characters and other problematic Unicode symbols
        filename = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270\U000E0000-\U000E007F]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        # Trim and limit length
        return filename.strip()[:100]
    
    def _trim_video_to_duration(self, video_path, target_duration):
        """Trim video to remove trailing frames and match actual duration"""
        try:
            if not target_duration or target_duration <= 0:
                return video_path
                
            # Check actual video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"📊 Video duration: YouTube={target_duration:.1f}s, Actual={actual_duration:.1f}s")
            
            # For squash videos, if the video is much longer than expected, 
            # try to find the actual match content
            duration_diff = actual_duration - target_duration
            
            # If the video is substantially longer, try to trim it intelligently
            if duration_diff > 300:  # More than 5 minutes difference
                print(f"⚠️ Large duration difference ({duration_diff:.1f}s), attempting smart trimming...")
                
                # For squash match videos, try to detect actual match content
                # by looking for consistent court activity
                smart_duration = self._detect_match_duration(video_path, fps)
                
                if smart_duration and smart_duration < actual_duration:
                    print(f"🎯 Detected actual match duration: {smart_duration:.1f}s")
                    target_duration = smart_duration
                else:
                    # Fallback: assume match is roughly 90 minutes max for squash
                    max_squash_duration = 90 * 60  # 90 minutes
                    if actual_duration > max_squash_duration:
                        print(f"⚠️ Video too long ({actual_duration/60:.1f}min), trimming to {max_squash_duration/60:.1f}min")
                        target_duration = max_squash_duration
                    else:
                        print(f"✅ Video duration reasonable, keeping original")
                        return video_path
            
            # Trim if necessary
            if actual_duration > target_duration + 30:  # Add 30s buffer
                print(f"✂️ Trimming video to {target_duration:.1f}s...")
                
                # Create trimmed filename
                base_name = os.path.splitext(video_path)[0]
                trimmed_path = f"{base_name}_trimmed.mp4"
                
                # Use OpenCV to trim video
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create output video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(trimmed_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print("⚠️ Could not create trimmed video, using original")
                    return video_path
                
                # Copy frames up to target duration
                target_frames = int(target_duration * fps)
                frames_written = 0
                
                while frames_written < target_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    out.write(frame)
                    frames_written += 1
                
                cap.release()
                out.release()
                
                # Verify trimmed video
                if os.path.exists(trimmed_path):
                    # Remove original and rename trimmed
                    os.remove(video_path)
                    os.rename(trimmed_path, video_path)
                    print(f"✅ Video trimmed to {target_duration:.1f}s")
                else:
                    print("⚠️ Trimming failed, using original video")
            
            return video_path
            
        except Exception as e:
            print(f"⚠️ Video trimming error: {e}, using original video")
            return video_path
    
    def _detect_match_duration(self, video_path, fps):
        """Detect actual match duration by analyzing court activity"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample every 30 seconds to detect court activity
            sample_interval = int(30 * fps)
            court_activity_frames = []
            
            print(f"🔍 Analyzing video for court activity...")
            
            for frame_idx in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if this frame shows court activity
                if self._is_court_activity_frame(frame):
                    court_activity_frames.append(frame_idx)
            
            cap.release()
            
            if len(court_activity_frames) < 2:
                return None
            
            # Find the start and end of consistent court activity
            # Look for the longest continuous period of court activity
            start_frame = court_activity_frames[0]
            end_frame = court_activity_frames[-1]
            
            # Convert to duration
            match_duration = end_frame / fps
            
            print(f"🎯 Court activity detected from {start_frame/fps:.1f}s to {end_frame/fps:.1f}s")
            return match_duration
            
        except Exception as e:
            print(f"⚠️ Match duration detection failed: {e}")
            return None
    
    def _is_court_activity_frame(self, frame):
        """Check if frame shows court activity (simplified court detection)"""
        try:
            # Convert to HSV for better analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Look for court-like colors (light surfaces)
            lower_court = np.array([0, 0, 150])
            upper_court = np.array([180, 50, 255])
            court_mask = cv2.inRange(hsv, lower_court, upper_court)
            court_percentage = np.sum(court_mask > 0) / (frame.shape[0] * frame.shape[1])
            
            # Court frames should have significant light-colored areas
            if court_percentage > 0.25:  # At least 25% light colors
                return True
            
            return False
            
        except Exception as e:
            return False

class FileManager:
    """Handles file management and utilities"""
    
    def __init__(self):
        pass
    
    def sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        # Remove invalid characters AND Unicode pipe characters
        filename = re.sub(r'[<>:"/\\|?*｜⧸]', '', filename)
        # Remove emoji characters and other problematic Unicode symbols
        filename = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270\U000E0000-\U000E007F]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        # Trim and limit length
        return filename.strip()[:100]
    
    def video_exists(self, video_path):
        """Check if a video file exists and is readable"""
        if not os.path.exists(video_path):
            return False
        
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except:
            return False
    
    def create_video_writer(self, output_path, fps, width, height):
        """Create video writer with fallback codec options"""
        # Try different codecs in order of preference
        codecs = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V - widely compatible
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID - good fallback
            cv2.VideoWriter_fourcc(*'MJPG'),  # MJPG - basic fallback
        ]
        
        for codec in codecs:
            try:
                out = cv2.VideoWriter(output_path, codec, fps, (width, height))
                if out.isOpened():
                    # Test write a blank frame
                    test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(test_frame)
                    return out
                out.release()
            except Exception as e:
                print(f"⚠️ Codec failed: {codec}, trying next...")
                continue
        
        print(f"❌ All codecs failed for {output_path}")
        return None
    
    def setup_directories(self, base_dir):
        """Set up directory structure for pipeline"""
        folders = {
            'downloads': os.path.join(base_dir, "01_Downloads"),
            'camera_segments': os.path.join(base_dir, "02_Camera_Segments"), 
            'player_annotations': os.path.join(base_dir, "03_Player_Annotations"),
            'final_outputs': os.path.join(base_dir, "04_Final_Outputs"),
            'logs': os.path.join(base_dir, "logs")
        }
        
        for folder_path in folders.values():
            os.makedirs(folder_path, exist_ok=True)
        
        print(f"📁 Created pipeline folder structure in: {base_dir}")
        return folders
    
    def get_file_size(self, file_path):
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    def check_disk_space(self, path, required_gb=1):
        """Check available disk space"""
        try:
            statvfs = os.statvfs(path)
            available_gb = (statvfs.f_frsize * statvfs.f_free) / (1024**3)
            return available_gb >= required_gb
        except:
            return True  # Assume space is available if check fails

class DataExporter:
    """Handles data collection and export functionality"""
    
    def __init__(self):
        pass
    
    def collect_frame_data(self, frame_idx, timestamp, player_data, video_title):
        """Collect frame data for CSV export"""
        # Base frame data
        frame_data = {
            'frame_number': frame_idx,
            'timestamp': timestamp,
            'video_title': video_title,
            'processing_time': datetime.now().isoformat()
        }
        
        # Add player data
        for i, player in enumerate(player_data[:2]):  # Max 2 players
            player_key = f"player_{i+1}"
            
            # Basic info
            frame_data[f"{player_key}_detected"] = True
            frame_data[f"{player_key}_label"] = player.get('label', '')
            frame_data[f"{player_key}_confidence"] = player.get('confidence', 0)
            
            # Bounding box
            bbox = player.get('bbox')
            if bbox:
                frame_data[f"{player_key}_bbox_x1"] = bbox[0]
                frame_data[f"{player_key}_bbox_y1"] = bbox[1]
                frame_data[f"{player_key}_bbox_x2"] = bbox[2]
                frame_data[f"{player_key}_bbox_y2"] = bbox[3]
                frame_data[f"{player_key}_bbox_center_x"] = (bbox[0] + bbox[2]) / 2
                frame_data[f"{player_key}_bbox_center_y"] = (bbox[1] + bbox[3]) / 2
            else:
                frame_data[f"{player_key}_bbox_x1"] = None
                frame_data[f"{player_key}_bbox_y1"] = None
                frame_data[f"{player_key}_bbox_x2"] = None
                frame_data[f"{player_key}_bbox_y2"] = None
                frame_data[f"{player_key}_bbox_center_x"] = None
                frame_data[f"{player_key}_bbox_center_y"] = None
            
            # Court position
            court_pos = player.get('court_position')
            if court_pos:
                frame_data[f"{player_key}_court_x"] = court_pos[0]
                frame_data[f"{player_key}_court_y"] = court_pos[1]
            else:
                frame_data[f"{player_key}_court_x"] = None
                frame_data[f"{player_key}_court_y"] = None
            
            # Pose landmarks
            pose = player.get('pose')
            if pose and hasattr(pose, 'pose_landmarks') and pose.pose_landmarks:
                landmarks = pose.pose_landmarks.landmark
                
                # Key body points for squash analysis
                key_points = {
                    'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
                    'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
                    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
                    'left_ankle': 27, 'right_ankle': 28
                }
                
                for point_name, landmark_idx in key_points.items():
                    if landmark_idx < len(landmarks):
                        landmark = landmarks[landmark_idx]
                        frame_data[f"{player_key}_{point_name}_x"] = landmark.x
                        frame_data[f"{player_key}_{point_name}_y"] = landmark.y
                        frame_data[f"{player_key}_{point_name}_z"] = landmark.z
                        frame_data[f"{player_key}_{point_name}_visibility"] = landmark.visibility
                    else:
                        frame_data[f"{player_key}_{point_name}_x"] = None
                        frame_data[f"{player_key}_{point_name}_y"] = None
                        frame_data[f"{player_key}_{point_name}_z"] = None
                        frame_data[f"{player_key}_{point_name}_visibility"] = None
            else:
                # No pose detected - fill with None
                key_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                             'right_knee', 'left_ankle', 'right_ankle']
                for point_name in key_points:
                    frame_data[f"{player_key}_{point_name}_x"] = None
                    frame_data[f"{player_key}_{point_name}_y"] = None
                    frame_data[f"{player_key}_{point_name}_z"] = None
                    frame_data[f"{player_key}_{point_name}_visibility"] = None
        
        # Fill missing players with None values
        for i in range(len(player_data), 2):
            player_key = f"player_{i+1}"
            frame_data[f"{player_key}_detected"] = False
            frame_data[f"{player_key}_label"] = None
            frame_data[f"{player_key}_confidence"] = None
            
            # Empty bbox
            for field in ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']:
                frame_data[f"{player_key}_bbox_{field}"] = None
            
            # Empty court position
            frame_data[f"{player_key}_court_x"] = None
            frame_data[f"{player_key}_court_y"] = None
            
            # Empty pose landmarks
            key_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                         'right_knee', 'left_ankle', 'right_ankle']
            for point_name in key_points:
                for coord in ['x', 'y', 'z', 'visibility']:
                    frame_data[f"{player_key}_{point_name}_{coord}"] = None
        
        return frame_data
    
    def save_to_csv(self, data_list, output_dir, video_title):
        """Save collected data to CSV file"""
        try:
            if not data_list:
                print("⚠️ No data to save")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data_list)
            
            # Generate filename
            safe_title = self._sanitize_filename(video_title)
            csv_filename = f"{safe_title}_player_data.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"📊 Saved player data: {len(data_list)} records to {csv_filename}")
            return csv_path
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            return None
    
    def _sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        # Remove invalid characters AND Unicode pipe characters
        filename = re.sub(r'[<>:"/\\|?*｜⧸]', '', filename)
        # Remove emoji characters and other problematic Unicode symbols
        filename = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270\U000E0000-\U000E007F]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        # Trim and limit length
        return filename.strip()[:100]
    
    def add_rally_detection_data(self, frame_data, is_rally_active=None):
        """Add rally detection data to frame data"""
        frame_data['rally_active'] = is_rally_active
        frame_data['rally_intensity'] = None  # Can be added later
        return frame_data

def create_frame_difference_video(input_video_path, output_video_path=None, diff_threshold=30, max_duration=None):
    """
    Create a frame difference video showing movement between consecutive frames.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output difference video
        diff_threshold: Threshold for difference detection (0-255)
        max_duration: Maximum duration in seconds to process (None for full video)
    
    Returns:
        Path to created difference video
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_frame_diff' + ext
    
    print(f"🎬 Creating frame difference video...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate max frames to process
    if max_duration:
        max_frames = int(max_duration * fps)
        total_frames = min(total_frames, max_frames)
        print(f"   ⏱️ Processing {max_duration}s ({total_frames} frames)")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        return None
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        out.release()
        return None
    
    # Convert to grayscale for difference calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Write first frame as black (no previous frame to compare)
    black_frame = np.zeros_like(prev_frame)
    out.write(black_frame)
    
    frame_count = 1
    
    with tqdm(total=total_frames-1, desc="   Creating frame differences", unit="frame") as pbar:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max duration
            if max_duration and frame_count >= total_frames:
                break
            
            # Convert current frame to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold to highlight significant changes
            _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Create colored difference frame
            # White = movement, Black = no movement
            diff_colored = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
            
            # Optional: Add original frame overlay with low opacity
            alpha = 0.3
            diff_overlay = cv2.addWeighted(current_frame, alpha, diff_colored, 1-alpha, 0)
            
            # Write difference frame
            out.write(diff_overlay)
            
            # Update for next iteration
            prev_gray = current_gray
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"✅ Frame difference video created: {output_video_path}")
    return output_video_path

def create_motion_heatmap_video(input_video_path, output_video_path=None, history_frames=30, max_duration=None):
    """
    Create a motion heatmap video showing accumulated movement over time.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output heatmap video
        history_frames: Number of frames to accumulate for heatmap
        max_duration: Maximum duration in seconds to process (None for full video)
    
    Returns:
        Path to created heatmap video
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    from collections import deque
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_motion_heatmap' + ext
    
    print(f"🔥 Creating motion heatmap video...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate max frames to process
    if max_duration:
        max_frames = int(max_duration * fps)
        total_frames = min(total_frames, max_frames)
        print(f"   ⏱️ Processing {max_duration}s ({total_frames} frames)")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        return None
    
    # Initialize motion history buffer
    motion_history = deque(maxlen=history_frames)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        out.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize first frame
    motion_history.append(np.zeros((height, width), dtype=np.uint8))
    
    frame_count = 1
    
    with tqdm(total=total_frames-1, desc="   Creating motion heatmap", unit="frame") as pbar:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max duration
            if max_duration and frame_count >= total_frames:
                break
            
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Add to motion history
            motion_history.append(diff_thresh)
            
            # Create accumulated heatmap
            heatmap = np.zeros((height, width), dtype=np.float32)
            for i, motion_frame in enumerate(motion_history):
                # Weight recent frames more heavily
                weight = (i + 1) / len(motion_history)
                heatmap += motion_frame.astype(np.float32) * weight
            
            # Normalize heatmap
            heatmap = np.clip(heatmap / len(motion_history), 0, 255).astype(np.uint8)
            
            # Apply colormap (hot = more motion)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            
            # Overlay on original frame
            alpha = 0.4
            result = cv2.addWeighted(current_frame, 1-alpha, heatmap_colored, alpha, 0)
            
            # Write frame
            out.write(result)
            
            # Update for next iteration
            prev_gray = current_gray
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"✅ Motion heatmap video created: {output_video_path}")
    return output_video_path

def create_multi_duration_heatmap_grid(input_video_path, output_video_path=None, durations=[1, 2, 4, 8, 16, 32, 64, 128, 256], max_duration=None):
    """
    Create a 3x3 grid video showing motion heatmaps with different history durations.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output grid video
        durations: List of frame history durations to test
        max_duration: Maximum duration in seconds to process (None for full video)
    
    Returns:
        Path to created grid video
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    from collections import deque
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_heatmap_grid' + ext
    
    print(f"🔥 Creating multi-duration heatmap grid video...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    print(f"   🎯 Durations: {durations}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate max frames to process
    if max_duration:
        max_frames = int(max_duration * fps)
        total_frames = min(total_frames, max_frames)
        print(f"   ⏱️ Processing {max_duration}s ({total_frames} frames)")
    
    # Calculate grid dimensions
    grid_size = 3
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        return None
    
    # Initialize motion history buffers for each duration
    motion_histories = {}
    for duration in durations:
        motion_histories[duration] = deque(maxlen=duration)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        out.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize first frame for all durations
    for duration in durations:
        motion_histories[duration].append(np.zeros((height, width), dtype=np.uint8))
    
    frame_count = 1
    
    with tqdm(total=total_frames-1, desc="   Creating multi-duration heatmap grid", unit="frame") as pbar:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max duration
            if max_duration and frame_count >= total_frames:
                break
            
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Update all motion histories
            for duration in durations:
                motion_histories[duration].append(diff_thresh)
            
            # Create grid frame
            grid_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process each cell in the grid
            for i, duration in enumerate(durations):
                if i >= grid_size * grid_size:
                    break
                
                row = i // grid_size
                col = i % grid_size
                
                # Create heatmap for this duration
                heatmap = np.zeros((height, width), dtype=np.float32)
                history = motion_histories[duration]
                
                for j, motion_frame in enumerate(history):
                    # Weight recent frames more heavily
                    weight = (j + 1) / len(history)
                    heatmap += motion_frame.astype(np.float32) * weight
                
                # Normalize heatmap
                heatmap = np.clip(heatmap / len(history), 0, 255).astype(np.uint8)
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
                
                # Overlay on original frame
                alpha = 0.5
                cell_result = cv2.addWeighted(current_frame, 1-alpha, heatmap_colored, alpha, 0)
                
                # Resize to cell size
                cell_resized = cv2.resize(cell_result, (cell_width, cell_height))
                
                # Add label
                label = f"{duration}f"
                cv2.putText(cell_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Place in grid
                y1 = row * cell_height
                y2 = y1 + cell_height
                x1 = col * cell_width
                x2 = x1 + cell_width
                
                grid_frame[y1:y2, x1:x2] = cell_resized
            
            # Write grid frame
            out.write(grid_frame)
            
            # Update for next iteration
            prev_gray = current_gray
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"✅ Multi-duration heatmap grid created: {output_video_path}")
    return output_video_path

def create_frame_diff_comparison_grid(input_video_path, output_video_path=None, thresholds=[10, 20, 30, 40, 50, 60, 70, 80, 90], max_duration=None):
    """
    Create a 3x3 grid video showing frame differences with different thresholds.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output grid video
        thresholds: List of difference thresholds to test
        max_duration: Maximum duration in seconds to process (None for full video)
    
    Returns:
        Path to created grid video
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_diff_grid' + ext
    
    print(f"🎬 Creating frame difference threshold grid video...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    print(f"   🎯 Thresholds: {thresholds}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate max frames to process
    if max_duration:
        max_frames = int(max_duration * fps)
        total_frames = min(total_frames, max_frames)
        print(f"   ⏱️ Processing {max_duration}s ({total_frames} frames)")
    
    # Calculate grid dimensions
    grid_size = 3
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        return None
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        out.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Write first frame as black grid
    grid_frame = np.zeros((height, width, 3), dtype=np.uint8)
    out.write(grid_frame)
    
    frame_count = 1
    
    with tqdm(total=total_frames-1, desc="   Creating frame difference threshold grid", unit="frame") as pbar:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max duration
            if max_duration and frame_count >= total_frames:
                break
            
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Create grid frame
            grid_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Process each cell in the grid
            for i, threshold in enumerate(thresholds):
                if i >= grid_size * grid_size:
                    break
                
                row = i // grid_size
                col = i % grid_size
                
                # Apply threshold
                _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                
                # Create colored difference
                diff_colored = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
                
                # Overlay on original frame
                alpha = 0.3
                cell_result = cv2.addWeighted(current_frame, alpha, diff_colored, 1-alpha, 0)
                
                # Resize to cell size
                cell_resized = cv2.resize(cell_result, (cell_width, cell_height))
                
                # Add label
                label = f"T{threshold}"
                cv2.putText(cell_resized, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Place in grid
                y1 = row * cell_height
                y2 = y1 + cell_height
                x1 = col * cell_width
                x2 = x1 + cell_width
                
                grid_frame[y1:y2, x1:x2] = cell_resized
            
            # Write grid frame
            out.write(grid_frame)
            
            # Update for next iteration
            prev_gray = current_gray
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"✅ Frame difference threshold grid created: {output_video_path}")
    return output_video_path

def detect_players_and_ball_from_motion(input_video_path, output_video_path=None, diff_threshold=10, max_duration=None):
    """
    Detect players and ball using frame difference analysis.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video with detections
        diff_threshold: Threshold for motion detection (lower = more sensitive)
        max_duration: Maximum duration in seconds to process
    
    Returns:
        Dict with detection results and output video path
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_motion_detection' + ext
    
    print(f"🎯 Detecting players and ball from motion...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    print(f"   🎯 Threshold: {diff_threshold}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate max frames to process
    if max_duration:
        max_frames = int(max_duration * fps)
        total_frames = min(total_frames, max_frames)
        print(f"   ⏱️ Processing {max_duration}s ({total_frames} frames)")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        return None
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        out.release()
        return None
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detection results storage
    player_detections = []
    ball_detections = []
    
    # Write first frame
    out.write(prev_frame)
    
    frame_count = 1
    
    with tqdm(total=total_frames-1, desc="   Detecting motion", unit="frame") as pbar:
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max duration
            if max_duration and frame_count >= total_frames:
                break
            
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, current_gray)
            
            # Apply threshold
            _, diff_thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Clean up noise
            kernel = np.ones((3,3), np.uint8)
            diff_clean = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
            diff_clean = cv2.morphologyEx(diff_clean, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect players and ball
            frame_players = detect_players_in_frame(contours, current_frame, frame_count)
            frame_ball = detect_ball_in_frame(contours, current_frame, prev_frame, frame_count)
            
            # Store detections
            player_detections.append(frame_players)
            ball_detections.append(frame_ball)
            
            # Draw detections on frame
            result_frame = draw_detections(current_frame, frame_players, frame_ball)
            
            # Write frame
            out.write(result_frame)
            
            # Update for next iteration
            prev_gray = current_gray
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    # Compile results
    results = {
        'output_video': output_video_path,
        'player_detections': player_detections,
        'ball_detections': ball_detections,
        'total_frames': frame_count - 1,
        'fps': fps
    }
    
    print(f"✅ Motion detection completed: {output_video_path}")
    return results

def detect_players_in_frame(contours, frame, frame_idx):
    """
    Detect players from motion contours in a single frame.
    
    Args:
        contours: OpenCV contours from frame difference
        frame: Current frame (for color analysis)
        frame_idx: Frame number
    
    Returns:
        List of player detections
    """
    players = []
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        
        # Filter by area (players are large moving objects)
        if area < 1000 or area > 50000:  # Adjust based on video resolution
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (players are roughly vertical)
        aspect_ratio = h / w if w > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 4.0:  # Too wide or too narrow
            continue
        
        # Filter by dimensions (reasonable player size)
        if w < 20 or h < 50 or w > 200 or h > 400:
            continue
        
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Player confidence based on size and aspect ratio
        confidence = min(1.0, area / 10000) * min(1.0, aspect_ratio / 2.0)
        
        players.append({
            'bbox': [x, y, x+w, y+h],
            'center': [center_x, center_y],
            'area': area,
            'confidence': confidence,
            'frame': frame_idx
        })
    
    # Sort by confidence and return top 2 (max 2 players)
    players.sort(key=lambda p: p['confidence'], reverse=True)
    return players[:2]

def detect_ball_in_frame(contours, current_frame, prev_frame, frame_idx):
    """
    Detect ball from motion contours in a single frame.
    
    Args:
        contours: OpenCV contours from frame difference
        current_frame: Current frame
        prev_frame: Previous frame
        frame_idx: Frame number
    
    Returns:
        Ball detection or None
    """
    ball_candidates = []
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        
        # Filter by area (ball is small)
        if area < 5 or area > 500:  # Very small object
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by dimensions (ball is roughly circular/small)
        if w < 3 or h < 3 or w > 30 or h > 30:
            continue
        
        # Check aspect ratio (ball should be roughly square)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
        if aspect_ratio > 2.0:  # Too elongated
            continue
        
        # Calculate center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if this region is bright/white in the current frame
        roi = current_frame[y:y+h, x:x+w]
        if roi.size > 0:
            # Convert to grayscale for brightness check
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(roi_gray)
            
            # Ball should be bright (white)
            if avg_brightness < 150:  # Not bright enough
                continue
            
            # Check for white/bright color
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Look for low saturation (white/gray) and high value (bright)
            avg_saturation = np.mean(roi_hsv[:,:,1])
            avg_value = np.mean(roi_hsv[:,:,2])
            
            if avg_saturation > 100 or avg_value < 150:  # Not white/bright enough
                continue
        else:
            continue
        
        # Ball confidence based on size, brightness, and shape
        size_score = min(1.0, area / 100)  # Prefer medium-small objects
        brightness_score = min(1.0, avg_brightness / 255)
        shape_score = min(1.0, 2.0 / aspect_ratio)  # Prefer circular
        
        confidence = size_score * brightness_score * shape_score
        
        ball_candidates.append({
            'bbox': [x, y, x+w, y+h],
            'center': [center_x, center_y],
            'area': area,
            'confidence': confidence,
            'brightness': avg_brightness,
            'frame': frame_idx
        })
    
    # Return the most confident ball detection
    if ball_candidates:
        ball_candidates.sort(key=lambda b: b['confidence'], reverse=True)
        return ball_candidates[0]
    
    return None

def draw_detections(frame, players, ball):
    """
    Draw player and ball detections on the frame.
    
    Args:
        frame: Input frame
        players: List of player detections
        ball: Ball detection or None
    
    Returns:
        Frame with detections drawn
    """
    result = frame.copy()
    
    # Draw players
    for i, player in enumerate(players):
        bbox = player['bbox']
        center = player['center']
        confidence = player['confidence']
        
        # Player colors
        color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for P1, Red for P2
        
        # Draw bounding box
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw center point
        cv2.circle(result, tuple(center), 5, color, -1)
        
        # Draw label
        label = f"P{i+1} {confidence:.2f}"
        cv2.putText(result, label, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw ball
    if ball:
        bbox = ball['bbox']
        center = ball['center']
        confidence = ball['confidence']
        
        # Ball color (yellow)
        color = (0, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw center point
        cv2.circle(result, tuple(center), 3, color, -1)
        
        # Draw label
        label = f"Ball {confidence:.2f}"
        cv2.putText(result, label, (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result

def create_frame_diff_and_detect_players(input_video_path, output_video_path=None, diff_threshold=10, max_duration=None):
    """
    Create frame difference video and run YOLO human detection on it.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video with detections
        diff_threshold: Threshold for frame difference (lower = more sensitive)
        max_duration: Maximum duration in seconds to process
    
    Returns:
        Dict with detection results and output video path
    """
    import cv2
    import numpy as np
    from tqdm import tqdm
    import tempfile
    import os
    
    if output_video_path is None:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = base + '_framediff_detection' + ext
    
    print(f"🎯 Creating frame difference video and detecting players...")
    print(f"   📹 Input: {os.path.basename(input_video_path)}")
    print(f"   📹 Output: {os.path.basename(output_video_path)}")
    print(f"   🎯 Threshold: {diff_threshold}")
    
    # Step 1: Create frame difference video
    print(f"   🎬 Step 1: Creating frame difference video...")
    
    # Create temporary frame difference video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        temp_diff_video = tmp_file.name
    
    # Generate frame difference video
    diff_result = create_frame_difference_video(
        input_video_path, 
        temp_diff_video, 
        diff_threshold=diff_threshold,
        max_duration=max_duration
    )
    
    if not diff_result:
        print("❌ Failed to create frame difference video")
        return None
    
    # Step 2: Run YOLO detection on frame difference video
    print(f"   🤸 Step 2: Running YOLO detection on frame difference video...")
    
    # Import the player detector
    from player_detection import PlayerDetector
    
    # Initialize player detector
    player_detector = PlayerDetector()
    
    # Open the frame difference video
    cap = cv2.VideoCapture(temp_diff_video)
    if not cap.isOpened():
        print(f"❌ Could not open frame difference video: {temp_diff_video}")
        os.unlink(temp_diff_video)
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Could not create output video: {output_video_path}")
        cap.release()
        os.unlink(temp_diff_video)
        return None
    
    # Process frames and detect players
    player_detections = []
    frame_count = 0
    
    with tqdm(total=total_frames, desc="   Processing frame difference detection", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run player detection on frame difference frame
            try:
                annotated_frame, player_data, _ = player_detector.process_frame(
                    frame, 
                    frame_idx=frame_count,
                    crop_margin_percent=0.0  # No cropping needed for frame diff
                )
                
                # Store detection results
                player_detections.append({
                    'frame': frame_count,
                    'players': player_data,
                    'timestamp': frame_count / fps
                })
                
                # Write annotated frame
                out.write(annotated_frame)
                
            except Exception as e:
                print(f"⚠️ Error processing frame {frame_count}: {e}")
                # Write original frame if detection fails
                out.write(frame)
                player_detections.append({
                    'frame': frame_count,
                    'players': [],
                    'timestamp': frame_count / fps
                })
            
            frame_count += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    # Remove temporary frame difference video
    os.unlink(temp_diff_video)
    
    # Compile results
    results = {
        'output_video': output_video_path,
        'player_detections': player_detections,
        'total_frames': frame_count,
        'fps': fps,
        'diff_threshold': diff_threshold
    }
    
    # Calculate statistics
    frames_with_players = sum(1 for detection in player_detections if detection['players'])
    total_player_detections = sum(len(detection['players']) for detection in player_detections)
    
    print(f"   📊 Detection Statistics:")
    print(f"      Total frames processed: {frame_count}")
    print(f"      Frames with players: {frames_with_players} ({frames_with_players/frame_count*100:.1f}%)")
    print(f"      Total player detections: {total_player_detections}")
    print(f"      Avg players per frame: {total_player_detections/frame_count:.2f}")
    
    print(f"✅ Frame difference detection completed: {output_video_path}")
    return results

def extract_audio_from_video(video_path, audio_output_path=None):
    """Extract audio from video and save as .wav file. Returns path to .wav file."""
    import os
    import glob
    import subprocess
    if audio_output_path is None:
        base, _ = os.path.splitext(video_path)
        audio_output_path = base + '.wav'
    
    # First check if we have a separate audio file from the download process
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Look for separate audio files (from our download process)
    # Remove .f399 suffix if present (from camera processing)
    search_name = video_name
    if search_name.endswith('.f399'):
        search_name = search_name[:-5]
    # Remove enhanced_ prefix if present
    if search_name.startswith('enhanced_'):
        search_name = search_name[9:]
    # Remove shell escaping from filename
    search_name = search_name.replace('\\!', '!')
    
    audio_patterns = [
        os.path.join(video_dir, f"{search_name}_audio.*"),
        os.path.join(video_dir, f"*_audio.*")
    ]
    
    # Also look in the downloads folder if we're processing a processed video
    if '/03_Player_Annotations/' in video_path or '/04_Final_Outputs/' in video_path:
        # Go back to the downloads folder
        base_output_dir = video_path.split('/03_Player_Annotations/')[0] if '/03_Player_Annotations/' in video_path else video_path.split('/04_Final_Outputs/')[0]
        downloads_dir = os.path.join(base_output_dir, '01_Downloads')
        if os.path.exists(downloads_dir):
            audio_patterns.extend([
                os.path.join(downloads_dir, '**', f"{search_name}_audio.*"),
                os.path.join(downloads_dir, '**', f"{search_name}.*")
            ])
    
    separate_audio_file = None
    for pattern in audio_patterns:
        audio_files = glob.glob(pattern, recursive=True)
        if audio_files:
            separate_audio_file = audio_files[0]
            break
    
    if separate_audio_file:
        print(f"🔊 Using separate audio file: {os.path.basename(separate_audio_file)}")
        try:
            import librosa
            # Load audio directly from the separate audio file
            y, sr = librosa.load(separate_audio_file, sr=44100, mono=True)
            # Save as wav using soundfile
            sf.write(audio_output_path, y, sr)
            return audio_output_path
        except Exception as e:
            print(f"❌ Failed to process separate audio file: {e}")
    
    # Fallback: try to extract from video file
    print(f"🔊 Extracting audio from video file...")
    
    # First try with ffmpeg if available
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', audio_output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffmpeg is not available, try using librosa directly on the video
        try:
            import librosa
            # Load audio directly from video file using librosa
            y, sr = librosa.load(video_path, sr=44100, mono=True)
            # Save as wav using soundfile
            sf.write(audio_output_path, y, sr)
            return audio_output_path
        except Exception as e:
            print(f"❌ Audio extraction failed with both ffmpeg and librosa: {e}")
            return None