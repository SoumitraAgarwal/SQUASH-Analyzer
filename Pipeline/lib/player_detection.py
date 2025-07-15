"""
Player Detection Library
Complete player detection, pose estimation, and court mapping
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class PlayerDetector:
    """Complete player detection and pose estimation"""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize player detector with YOLO and MediaPipe"""
        try:
            # Load YOLO model
            model_file = os.path.join(os.path.dirname(__file__), model_path)
            self.yolo = YOLO(model_file)
            
            # Initialize MediaPipe pose
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Court mapper
            self.court_mapper = SquashCourtMapper()
            
            print("✅ Player detector initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize player detector: {e}")
            raise
    
    def process_frame(self, frame, show_boundaries=False):
        """Process frame for player detection and pose estimation"""
        try:
            # 1. YOLO detection
            results = self.yolo(frame, verbose=False)
            
            # 2. Extract player detections
            player_data = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Only process 'person' class (class 0)
                        if int(box.cls[0]) == 0:
                            confidence = float(box.conf[0])
                            if confidence > 0.5:
                                # Get bounding box
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                bbox = [int(x1), int(y1), int(x2), int(y2)]
                                
                                # Extract person region for pose estimation
                                person_region = frame[int(y1):int(y2), int(x1):int(x2)]
                                
                                # Get pose landmarks
                                pose_landmarks = self._get_pose_landmarks(person_region)
                                
                                # Map to court coordinates
                                court_pos = self.court_mapper.image_to_court(
                                    (x1 + x2) / 2, y2  # Center x, bottom y
                                )
                                
                                player_data.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'pose': pose_landmarks,
                                    'court_position': court_pos,
                                    'label': f'Player {len(player_data) + 1}'
                                })
            
            # 3. Draw annotations
            annotated_frame = self._draw_annotations(frame, player_data)
            
            return annotated_frame, player_data
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return frame, []
    
    def _get_pose_landmarks(self, person_region):
        """Extract pose landmarks from person region"""
        try:
            if person_region.size == 0:
                return None
                
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_image)
            
            return results if results.pose_landmarks else None
            
        except Exception as e:
            return None
    
    def _draw_annotations(self, frame, player_data):
        """Draw player annotations on frame"""
        try:
            annotated_frame = frame.copy()
            
            for i, player in enumerate(player_data):
                bbox = player['bbox']
                confidence = player['confidence']
                pose = player['pose']
                
                # Draw bounding box
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for player 1, red for player 2
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw label
                label = f"{player['label']} ({confidence:.2f})"
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw pose stick figure
                if pose and pose.pose_landmarks:
                    annotated_frame = self._draw_stick_figure(annotated_frame, pose.pose_landmarks, bbox)
            
            return annotated_frame
            
        except Exception as e:
            print(f"❌ Annotation drawing error: {e}")
            return frame
    
    def _draw_stick_figure(self, frame, landmarks, bbox):
        """Draw stick figure pose on frame"""
        try:
            # Adjust landmarks to frame coordinates
            frame_height, frame_width = frame.shape[:2]
            bbox_x, bbox_y, bbox_w, bbox_h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Key connections for stick figure
            connections = [
                (11, 12),  # Shoulders
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 23), (12, 24),  # Torso
                (23, 24),  # Hips
                (23, 25), (25, 27),  # Left leg
                (24, 26), (26, 28),  # Right leg
            ]
            
            # Draw connections
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                    start_landmark = landmarks.landmark[start_idx]
                    end_landmark = landmarks.landmark[end_idx]
                    
                    # Convert to frame coordinates
                    start_x = int(bbox_x + start_landmark.x * bbox_w)
                    start_y = int(bbox_y + start_landmark.y * bbox_h)
                    end_x = int(bbox_x + end_landmark.x * bbox_w)
                    end_y = int(bbox_y + end_landmark.y * bbox_h)
                    
                    # Draw line
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            return frame

class SquashCourtMapper:
    """Maps image coordinates to court coordinates"""
    
    def __init__(self):
        # Standard squash court dimensions (meters)
        self.court_width = 6.4
        self.court_length = 9.75
        
        # Simple mapping - in full version this would use homography
        self.image_width = 1920  # Assume 1080p
        self.image_height = 1080
    
    def image_to_court(self, x, y):
        """Convert image coordinates to court coordinates"""
        try:
            # Simple linear mapping
            court_x = (x / self.image_width) * self.court_width
            court_y = ((self.image_height - y) / self.image_height) * self.court_length
            
            return [court_x, court_y]
            
        except Exception as e:
            return [0, 0]
    
    def court_to_image(self, court_x, court_y):
        """Convert court coordinates to image coordinates"""
        try:
            x = (court_x / self.court_width) * self.image_width
            y = self.image_height - ((court_y / self.court_length) * self.image_height)
            
            return [int(x), int(y)]
            
        except Exception as e:
            return [0, 0]

class CourtVisualizer:
    """Creates court overlay visualizations"""
    
    def __init__(self):
        self.court_mapper = SquashCourtMapper()
    
    def create_court_overlay(self, player_data, overlay_size=200):
        """Create court overlay with player positions"""
        try:
            overlay = np.ones((overlay_size, overlay_size, 3), dtype=np.uint8) * 255
            
            # Draw court boundaries
            court_color = (200, 200, 200)
            line_thickness = 2
            
            # Court outline
            margin = 20
            court_width = overlay_size - 2 * margin
            court_height = int(court_width * 6.4 / 9.75)
            
            court_x = (overlay_size - court_width) // 2
            court_y = (overlay_size - court_height) // 2
            
            # Draw court
            cv2.rectangle(overlay, (court_x, court_y), 
                         (court_x + court_width, court_y + court_height), 
                         court_color, line_thickness)
            
            # Draw service lines
            service_line_y = court_y + court_height // 3
            t_line_x = court_x + court_width // 2
            
            cv2.line(overlay, (court_x, service_line_y), 
                    (court_x + court_width, service_line_y), court_color, 1)
            cv2.line(overlay, (t_line_x, service_line_y), 
                    (t_line_x, court_y + court_height), court_color, 1)
            
            # Draw players
            player_colors = [(0, 0, 255), (255, 0, 0)]
            for i, player in enumerate(player_data[:2]):
                court_pos = player.get('court_position', [0, 0])
                
                # Map to overlay coordinates
                overlay_x = int(court_x + (court_pos[0] / 6.4) * court_width)
                overlay_y = int(court_y + court_height - (court_pos[1] / 9.75) * court_height)
                
                # Draw player dot
                cv2.circle(overlay, (overlay_x, overlay_y), 5, player_colors[i], -1)
                
                # Add label
                label = player.get('label', f'P{i+1}')
                cv2.putText(overlay, label, (overlay_x + 8, overlay_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, player_colors[i], 1)
            
            return overlay
            
        except Exception as e:
            print(f"❌ Court overlay error: {e}")
            return np.ones((overlay_size, overlay_size, 3), dtype=np.uint8) * 255
    
    def add_overlay_to_frame(self, frame, overlay):
        """Add court overlay to video frame"""
        try:
            frame_height, frame_width = frame.shape[:2]
            overlay_height, overlay_width = overlay.shape[:2]
            
            # Position in top-right corner
            margin = 20
            y_start = margin
            x_start = frame_width - overlay_width - margin
            
            if x_start >= 0 and y_start + overlay_height <= frame_height:
                frame[y_start:y_start + overlay_height, x_start:x_start + overlay_width] = overlay
            
            return frame
            
        except Exception as e:
            return frame