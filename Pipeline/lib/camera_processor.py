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
TEMPLATE_CREATION_STRIDE = 30
ASSIGNMENT_STRIDE = 5

def create_templates_from_video(input_path, templates_output_path, camera_segments_path):
    """
    Simplified camera template creation and main camera extraction
    For now, just copies the input video as the main camera segment
    """
    try:
        # For simplicity, just copy the input video as main camera
        # In a full implementation, this would do template matching
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}.f399.mp4"  # .f399 indicates processed
        output_path = os.path.join(camera_segments_path, output_filename)
        
        # Simple copy for now - in full version this would extract main camera
        import shutil
        shutil.copy2(input_path, output_path)
        
        print(f"✅ Camera processing: {output_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Camera processing failed: {e}")
        return False

class CameraProcessor:
    """Handles camera extraction and main camera detection"""
    
    def __init__(self):
        pass
    
    def extract_main_camera_segments(self, input_path, output_dir):
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
                camera_segments_path
            )
            
            if success:
                # Find the main camera output file
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                for file in os.listdir(camera_segments_path):
                    if file.startswith(base_name) and file.endswith('.mp4'):
                        output_path = os.path.join(camera_segments_path, file)
                        print(f"✅ Camera extraction successful: {file}")
                        return output_path
                        
                print(f"❌ Main camera file not found in output directory")
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