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
            
            # yt-dlp options
            ydl_opts = {
                'outtmpl': os.path.join(folder_path, '%(title)s.%(ext)s'),
                'format': 'best[ext=mp4]/best',  # Prefer mp4
                'quiet': True,
                'no_warnings': True,
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                print(f"📹 Downloading: {title}")
                
                # Download
                ydl.download([video_url])
                
                # Find the downloaded file
                safe_title = self._sanitize_filename(title)
                for file in os.listdir(folder_path):
                    if file.startswith(safe_title) or title.split()[0] in file:
                        file_path = os.path.join(folder_path, file)
                        if os.path.exists(file_path):
                            return {
                                'title': title,
                                'file_path': file_path,
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
            available_gb = (statvfs.f_frsize * statvfs.f_availa) / (1024**3)
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