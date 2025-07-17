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
            
            # yt-dlp options - prioritize high quality without requiring ffmpeg
            ydl_opts = {
                'outtmpl': os.path.join(folder_path, '%(title)s.%(ext)s'),
                'format': (
                    # Try specific high quality formats first (no audio merging)
                    '299/298/136/'  # 1080p60, 720p60, 720p
                    'best[height>=1080][fps>=60]/best[height>=1080]/'  # Generic 1080p60, 1080p
                    'best[height>=720][fps>=60]/best[height>=720]/'    # Generic 720p60, 720p
                    'best'  # Final fallback
                ),
                'quiet': True,
                'no_warnings': True,
                'playliststart': 1,
                'playlistend': 1,
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', 'Unknown') if info else 'Unknown'
                duration = info.get('duration', 0) if info else 0
                
                # Find the best format that will be selected
                formats = info.get('formats', []) if info else []
                selected_format = None
                for fmt in formats:
                    if info and fmt.get('format_id') == info.get('format_id'):
                        selected_format = fmt
                        break
                
                if selected_format:
                    height = selected_format.get('height', 'Unknown')
                    fps = selected_format.get('fps', 'Unknown')
                    filesize = selected_format.get('filesize', 0)
                    filesize_mb = filesize / (1024*1024) if filesize else 0
                    
                    print(f"📹 Downloading: {title}")
                    print(f"🎬 Quality: {height}p @ {fps}fps ({filesize_mb:.1f}MB)")
                else:
                    print(f"📹 Downloading: {title}")
                
                # Download
                ydl.download([video_url])
                
                # Find the downloaded file
                safe_title = self._sanitize_filename(title)
                downloaded_file = None
                for file in os.listdir(folder_path):
                    if file.startswith(safe_title) or title.split()[0] in file:
                        file_path = os.path.join(folder_path, file)
                        if os.path.exists(file_path):
                            downloaded_file = file_path
                            break
                
                if downloaded_file:
                    # Trim video to actual duration to remove trailing frames
                    trimmed_file = self._trim_video_to_duration(downloaded_file, duration)
                    
                    return {
                        'title': title,
                        'file_path': trimmed_file,
                        'duration': duration
                    }
                
                # If exact match not found, return first video file
                for file in os.listdir(folder_path):
                    if file.endswith(('.mp4', '.mkv', '.avi', '.webm')):
                        return {
                            'title': title,
                            'file_path': os.path.join(folder_path, file),
                            'duration': duration
                        }
                
                print("❌ Downloaded file not found")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
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

def extract_audio_from_video(video_path, audio_output_path=None):
    """Extract audio from video and save as .wav file. Returns path to .wav file."""
    import os
    import tempfile
    import subprocess
    if audio_output_path is None:
        base, _ = os.path.splitext(video_path)
        audio_output_path = base + '.wav'
    
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