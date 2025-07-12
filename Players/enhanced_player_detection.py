import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from typing import List, Tuple, Optional

class EnhancedPlayerDetector:
    """
    Enhanced player detection with court boundary constraints and pose estimation.
    Combines YOLO for player detection with MediaPipe for pose estimation.
    """
    
    def __init__(self, yolo_model_path='yolov8n.pt'):
        # Initialize YOLO for player detection
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize MediaPipe for pose detection
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define court boundaries (as percentage of frame dimensions)
        # These define the valid area where players can be detected
        self.court_boundaries = {
            'min_y': 0.25,  # Top 25% is likely front wall/audience area
            'max_y': 0.95,  # Bottom 5% might be outside court
            'min_x': 0.05,  # Left 5% margin
            'max_x': 0.95   # Right 5% margin
        }
        
        # Pose connection colors for stick figure
        self.pose_colors = {
            'body': (0, 255, 0),      # Green for body
            'arms': (255, 0, 0),      # Red for arms
            'legs': (0, 0, 255),      # Blue for legs
            'face': (255, 255, 0)     # Yellow for face
        }
    
    def set_court_boundaries(self, min_y=0.25, max_y=0.95, min_x=0.05, max_x=0.95):
        """Set custom court boundaries as percentages of frame dimensions."""
        self.court_boundaries = {
            'min_y': min_y,
            'max_y': max_y,
            'min_x': min_x,
            'max_x': max_x
        }
    
    def is_within_court_boundaries(self, bbox, frame_width, frame_height):
        """Check if a bounding box is within the defined court boundaries."""
        x1, y1, x2, y2 = bbox
        
        # Calculate center point and bottom point of bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bottom_y = y2  # Feet position
        
        # Convert boundaries to pixel coordinates
        min_x = self.court_boundaries['min_x'] * frame_width
        max_x = self.court_boundaries['max_x'] * frame_width
        min_y = self.court_boundaries['min_y'] * frame_height
        max_y = self.court_boundaries['max_y'] * frame_height
        
        # Check if the player's feet are within court boundaries
        return (min_x <= center_x <= max_x and 
                min_y <= bottom_y <= max_y)
    
    def detect_players(self, frame):
        """Detect players in frame with court boundary filtering."""
        results = self.yolo_model(frame, verbose=False)
        player_boxes = []
        centroids = []
        
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # 0 is 'person' in COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2, y2)
                    
                    # Filter by court boundaries
                    if self.is_within_court_boundaries(bbox, frame_width, frame_height):
                        player_boxes.append(bbox)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        centroids.append((cx, cy))
        
        return player_boxes, centroids
    
    def detect_poses(self, frame, player_boxes):
        """Detect poses for each player bounding box."""
        poses = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for bbox in player_boxes:
            x1, y1, x2, y2 = bbox
            
            # Extract player region with some padding
            padding = 20
            x1_crop = max(0, x1 - padding)
            y1_crop = max(0, y1 - padding)
            x2_crop = min(frame.shape[1], x2 + padding)
            y2_crop = min(frame.shape[0], y2 + padding)
            
            # Crop player region
            player_crop = rgb_frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            if player_crop.size > 0:
                # Detect pose
                results = self.pose_detector.process(player_crop)
                
                if results.pose_landmarks:
                    # Convert normalized coordinates back to original frame coordinates
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        x = int(landmark.x * (x2_crop - x1_crop) + x1_crop)
                        y = int(landmark.y * (y2_crop - y1_crop) + y1_crop)
                        landmarks.append((x, y, landmark.visibility))
                    
                    poses.append({
                        'landmarks': landmarks,
                        'bbox': bbox,
                        'raw_landmarks': results.pose_landmarks
                    })
                else:
                    poses.append(None)
            else:
                poses.append(None)
        
        return poses
    
    def draw_stick_figure(self, frame, pose_data):
        """Draw stick figure representation of the pose."""
        if pose_data is None or pose_data['landmarks'] is None:
            return frame
        
        landmarks = pose_data['landmarks']
        
        # Define pose connections (MediaPipe pose model)
        connections = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),    # Face outline
            (0, 4), (4, 5), (5, 6), (6, 8),    # Face outline
            
            # Body
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            
            # Arms
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (15, 17), (15, 19), (15, 21),  # Left hand
            (16, 18), (16, 20), (16, 22),  # Right hand
            
            # Legs
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (27, 29), (27, 31),  # Left foot
            (28, 30), (28, 32),  # Right foot
        ]
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                landmarks[start_idx][2] > 0.5 and landmarks[end_idx][2] > 0.5):  # Visibility check
                
                start_point = (landmarks[start_idx][0], landmarks[start_idx][1])
                end_point = (landmarks[end_idx][0], landmarks[end_idx][1])
                
                # Choose color based on body part
                if connection in [(11, 12), (11, 23), (12, 24), (23, 24)]:
                    color = self.pose_colors['body']
                elif connection in [(11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22)]:
                    color = self.pose_colors['arms']
                elif connection in [(23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (27, 31), (28, 30), (28, 32)]:
                    color = self.pose_colors['legs']
                else:
                    color = self.pose_colors['face']
                
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw key joints
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Major joints
        for joint_idx in key_joints:
            if joint_idx < len(landmarks) and landmarks[joint_idx][2] > 0.5:
                point = (landmarks[joint_idx][0], landmarks[joint_idx][1])
                cv2.circle(frame, point, 3, (255, 255, 255), -1)
                cv2.circle(frame, point, 3, (0, 0, 0), 1)
        
        return frame
    
    def draw_court_boundaries(self, frame):
        """Draw court boundaries for debugging."""
        height, width = frame.shape[:2]
        
        # Calculate boundary coordinates
        min_x = int(self.court_boundaries['min_x'] * width)
        max_x = int(self.court_boundaries['max_x'] * width)
        min_y = int(self.court_boundaries['min_y'] * height)
        max_y = int(self.court_boundaries['max_y'] * height)
        
        # Draw boundary rectangle
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
        cv2.putText(frame, 'Valid Court Area', (min_x, min_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame, show_boundaries=False):
        """
        Process a frame with enhanced player and pose detection.
        
        Returns:
            annotated_frame: Frame with annotations
            player_data: List of player data with poses
        """
        # Detect players with court boundary filtering
        player_boxes, centroids = self.detect_players(frame)
        
        # Detect poses for each player
        poses = self.detect_poses(frame, player_boxes)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw court boundaries if requested
        if show_boundaries:
            annotated_frame = self.draw_court_boundaries(annotated_frame)
        
        # Draw player bounding boxes and poses
        player_data = []
        for i, (bbox, pose_data) in enumerate(zip(player_boxes, poses)):
            if i >= 2:  # Limit to 2 players
                break
                
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Player {i+1}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw stick figure pose
            if pose_data is not None:
                annotated_frame = self.draw_stick_figure(annotated_frame, pose_data)
            
            # Store player data
            player_data.append({
                'bbox': bbox,
                'centroid': centroids[i] if i < len(centroids) else None,
                'pose': pose_data,
                'label': f'Player {i+1}'
            })
        
        return annotated_frame, player_data


def test_enhanced_detection():
    """Test the enhanced player detection system."""
    import os
    
    # Find a test video
    video_dir = "../Camera/camera_outputs/main_camera_segments"
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} not found")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    # Use first video for testing
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Testing with: {video_files[0]}")
    
    # Initialize detector
    detector = EnhancedPlayerDetector()
    
    # Process a few frames
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while frame_count < 10:  # Process first 10 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop frame (same as in main processing)
        crop_margin = int(0.10 * frame.shape[1])
        cropped_frame = frame[:, crop_margin:frame.shape[1]-crop_margin]
        
        # Process frame
        annotated_frame, player_data = detector.process_frame(cropped_frame, show_boundaries=True)
        
        print(f"Frame {frame_count}: Detected {len(player_data)} players")
        for i, player in enumerate(player_data):
            has_pose = player['pose'] is not None
            print(f"  Player {i+1}: BBox={player['bbox']}, Pose={'Yes' if has_pose else 'No'}")
        
        # Save annotated frame for inspection
        output_path = f"test_enhanced_frame_{frame_count}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        
        frame_count += 1
    
    cap.release()
    print(f"\nTest completed. Saved {frame_count} annotated frames.")


if __name__ == "__main__":
    test_enhanced_detection()