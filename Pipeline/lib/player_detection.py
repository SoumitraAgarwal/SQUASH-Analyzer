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
                model_complexity=0,  # Reduced from 1 to 0 for faster processing
                enable_segmentation=False,
                min_detection_confidence=0.3,  # Reduced from 0.5 for faster processing
                min_tracking_confidence=0.3   # Reduced from 0.5 for faster processing
            )
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Court mapper
            self.court_mapper = SquashCourtMapper()
            
            # Player tracking
            self.previous_player_positions = []  # Store previous player centroids
            self.tracking_search_radius = 200  # Pixels to search around previous positions
            
            # Jersey color tracking for consistent player IDs
            self.player_colors = {}  # Store average jersey colors for each player
            self.color_tracking_enabled = True
            
            print("✅ Player detector initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize player detector: {e}")
            raise
    
    def process_frame(self, frame, show_boundaries=False, crop_margin_percent=0.15, frame_idx=0):
        """Process frame for player detection and pose estimation"""
        try:
            # 1. Apply side cropping to remove audience (for detection only)
            original_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            crop_margin = int(crop_margin_percent * frame_width)
            detection_frame = frame[:, crop_margin:frame_width-crop_margin]
            
            # 2. YOLO detection on cropped frame
            results = self.yolo(detection_frame, verbose=False)
            
            # Debug: Check if YOLO is working
            if frame_idx % 100 == 0:
                print(f"Debug: YOLO results on frame {frame_idx}: {len(results)} result objects")
            
            # 2. Extract all person detections
            detection_candidates = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Only process 'person' class (class 0)
                        if int(box.cls[0]) == 0:
                            confidence = float(box.conf[0])
                            if confidence > 0.20:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1 += crop_margin
                                x2 += crop_margin
                                bbox = [int(x1), int(y1), int(x2), int(y2)]
                                centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
                                detection_candidates.append({
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'centroid': centroid
                                })
            # 3. Filter out candidates in the front wall area
            player_candidates = []
            for candidate in detection_candidates:
                bbox = candidate['bbox']
                if not self._is_front_wall_detection(bbox, original_frame.shape):
                    x1, y1, x2, y2 = bbox
                    person_region = original_frame[int(y1):int(y2), int(x1):int(x2)]
                    pose_landmarks = self._get_pose_landmarks(person_region)
                    court_pos = self.court_mapper.image_to_court((x1 + x2) / 2, y2)
                    jersey_color = self._extract_jersey_color(original_frame, bbox)
                    player_candidates.append({
                        'bbox': bbox,
                        'confidence': candidate['confidence'],
                        'pose': pose_landmarks,
                        'court_position': court_pos,
                        'label': f'Player {len(player_candidates) + 1}',
                        'centroid': candidate['centroid'],
                        'jersey_color': jersey_color
                    })
            # 4. Sort by confidence and keep only the best two as players
            # Select top 2 players by confidence, but only if bbox area >= 15000
            min_area = 15000
            filtered_candidates = [c for c in player_candidates if (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]) >= min_area]
            filtered_candidates = sorted(filtered_candidates, key=lambda x: x['confidence'], reverse=True)
            player_data = filtered_candidates[:2]
            # 5. Update tracking positions
            if player_data:
                self.previous_player_positions = [p['centroid'] for p in player_data]
            # 6. Assign consistent player IDs based on jersey colors (every 30 frames)
            if self.color_tracking_enabled and player_data and frame_idx % 30 == 0:
                player_data = self._assign_consistent_player_ids(player_data)
            # 7. Draw annotations on the original frame (not cropped)
            # Pass all detections and mark which are players
            annotated_frame = self._draw_annotations(original_frame, player_data, detection_candidates)
            if frame_idx % 100 == 0:
                print(f"Debug: Found {len(player_data)} non-front-wall players on frame {frame_idx}")
            return annotated_frame, player_data, detection_candidates
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return frame, [], []
    
    def _sort_by_tracking_priority(self, detection_candidates):
        """Sort detection candidates by proximity to previous player positions"""
        try:
            # Calculate minimum distance to any previous player position
            def tracking_priority(candidate):
                centroid = candidate['centroid']
                min_distance = float('inf')
                
                for prev_pos in self.previous_player_positions:
                    distance = np.sqrt((centroid[0] - prev_pos[0])**2 + (centroid[1] - prev_pos[1])**2)
                    min_distance = min(min_distance, distance)
                
                # Combine distance and confidence for priority
                # Lower distance = higher priority, higher confidence = higher priority
                tracking_score = min_distance - (candidate['confidence'] * 100)
                return tracking_score
            
            # Sort by tracking priority (lower score = higher priority)
            detection_candidates.sort(key=tracking_priority)
            return detection_candidates
            
        except Exception as e:
            # Fall back to confidence sorting if tracking fails
            detection_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            return detection_candidates
    
    def _extract_jersey_color(self, frame, bbox):
        """Extract dominant jersey color from player's torso region"""
        try:
            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            
            # Focus on torso region (middle 40% horizontally, upper 60% vertically)
            torso_x1 = int(x1 + bbox_w * 0.3)
            torso_x2 = int(x1 + bbox_w * 0.7)
            torso_y1 = int(y1 + bbox_h * 0.2)
            torso_y2 = int(y1 + bbox_h * 0.8)
            
            # Extract torso region
            torso_region = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            
            if torso_region.size == 0:
                return None
            
            # Convert to HSV for better color analysis
            hsv_torso = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average color (ignore very dark/light pixels)
            mask = cv2.inRange(hsv_torso, np.array([0, 30, 30]), np.array([180, 255, 255]))
            
            if np.sum(mask) > 0:
                # Calculate mean color of non-masked pixels
                mean_color = cv2.mean(hsv_torso, mask=mask)
                return mean_color[:3]  # Return H, S, V values
            else:
                # Fallback to simple mean
                return np.mean(hsv_torso.reshape(-1, 3), axis=0)
                
        except Exception as e:
            return None
    
    def _calculate_color_similarity(self, color1, color2):
        """Calculate similarity between two HSV colors"""
        try:
            if color1 is None or color2 is None:
                return 0.0
            
            # Convert to numpy arrays
            c1 = np.array(color1)
            c2 = np.array(color2)
            
            # Calculate weighted distance in HSV space
            # Hue is circular, so handle it specially
            hue_diff = min(abs(c1[0] - c2[0]), 360 - abs(c1[0] - c2[0]))
            sat_diff = abs(c1[1] - c2[1])
            val_diff = abs(c1[2] - c2[2])
            
            # Weighted similarity (hue is most important for jersey color)
            similarity = 1.0 - (hue_diff / 180.0 * 0.6 + sat_diff / 255.0 * 0.3 + val_diff / 255.0 * 0.1)
            return max(0.0, similarity)
            
        except Exception as e:
            return 0.0
    
    def _assign_consistent_player_ids(self, player_data):
        """Assign consistent player IDs based on jersey colors"""
        try:
            if len(player_data) == 0:
                return player_data
            
            # If we don't have stored colors yet, initialize them
            if not self.player_colors:
                for i, player in enumerate(player_data):
                    if player['jersey_color'] is not None:
                        self.player_colors[f'Player {i+1}'] = player['jersey_color']
                        player['label'] = f'Player {i+1}'
                return player_data
            
            # Match current detections to stored player colors
            updated_player_data = []
            used_player_ids = set()
            
            for player in player_data:
                if player['jersey_color'] is None:
                    # If no color detected, assign based on position or use next available ID
                    available_ids = [f'Player {i+1}' for i in range(len(self.player_colors))]
                    for pid in available_ids:
                        if pid not in used_player_ids:
                            player['label'] = pid
                            used_player_ids.add(pid)
                            break
                    updated_player_data.append(player)
                    continue
                
                # Find best color match
                best_match = None
                best_similarity = 0.0
                
                for player_id, stored_color in self.player_colors.items():
                    if player_id not in used_player_ids:
                        similarity = self._calculate_color_similarity(player['jersey_color'], stored_color)
                        if similarity > best_similarity and similarity > 0.5:  # Lowered threshold from 0.7 to 0.5
                            best_similarity = similarity
                            best_match = player_id
                
                if best_match:
                    player['label'] = best_match
                    used_player_ids.add(best_match)
                    # Update stored color with weighted average
                    old_color = np.array(self.player_colors[best_match])
                    new_color = np.array(player['jersey_color'])
                    self.player_colors[best_match] = (old_color * 0.8 + new_color * 0.2).tolist()
                else:
                    # No good match found, create new player or assign to unused slot
                    available_ids = [f'Player {i+1}' for i in range(max(2, len(self.player_colors) + 1))]
                    for pid in available_ids:
                        if pid not in used_player_ids:
                            player['label'] = pid
                            used_player_ids.add(pid)
                            if pid not in self.player_colors:
                                self.player_colors[pid] = player['jersey_color']
                            break
                
                updated_player_data.append(player)
            
            # Sort by player ID to ensure consistent ordering
            updated_player_data.sort(key=lambda x: x['label'])
            
            return updated_player_data
            
        except Exception as e:
            # If color tracking fails, fall back to original labeling
            for i, player in enumerate(player_data):
                player['label'] = f'Player {i+1}'
            return player_data
    
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
    
    def _is_front_wall_detection(self, bbox, frame_shape):
        """Check if detection is in front wall area (cameramen, officials, etc.)"""
        try:
            frame_height, frame_width = frame_shape[:2]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            # Conservative front wall detection - only filter obvious cases
            bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
            frame_area = frame_width * frame_height
            area_ratio = bbox_area / frame_area
            
            # Only filter very small detections at the bottom edge (cameramen behind glass)
            bottom_margin = frame_height * 0.05  # Only bottom 5% of frame
            if (bbox_y2 > frame_height - bottom_margin and area_ratio < 0.01):
                return True  # Very small detection at extreme bottom edge
                
            return False
            
        except Exception as e:
            return False
    
    def _is_valid_court_area(self, bbox, frame_shape):
        """Check if detection is in the main court area where players should be"""
        try:
            frame_height, frame_width = frame_shape[:2]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            
            # Define the main court area (middle 70% horizontally, middle 60% vertically)
            # This excludes audience, officials, and front wall areas
            
            # Horizontal bounds (exclude far left/right edges)
            left_margin = frame_width * 0.15   # 15% from left
            right_margin = frame_width * 0.85  # 85% from left (15% from right)
            
            # Vertical bounds (exclude top audience and front wall)
            top_margin = frame_height * 0.20   # 20% from top (exclude audience)
            bottom_margin = frame_height * 0.75 # 75% from top (exclude front wall)
            
            # Check if detection center is in valid court area
            bbox_center_x = (bbox_x1 + bbox_x2) / 2
            bbox_center_y = (bbox_y1 + bbox_y2) / 2
            
            # Must be within court boundaries
            if (left_margin <= bbox_center_x <= right_margin and
                top_margin <= bbox_center_y <= bottom_margin):
                
                # Additional check: detection should have reasonable size for court players
                bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
                frame_area = frame_width * frame_height
                area_ratio = bbox_area / frame_area
                
                # Court players should be reasonably sized (not tiny dots or huge)
                if 0.005 <= area_ratio <= 0.3:  # Between 0.5% and 30% of frame
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _intelligent_player_pairing(self, player_data, frame_shape):
        """Intelligent player pairing to ensure we get 2 court players"""
        try:
            if len(player_data) == 0:
                return []
            
            # If we have exactly 2 players, return them
            if len(player_data) == 2:
                return player_data[:2]
            
            # If we have more than 2 players, select the best 2
            if len(player_data) > 2:
                return self._select_best_two_players(player_data, frame_shape)
            
            # If we have only 1 player, try to find the missing player
            if len(player_data) == 1:
                return self._find_missing_player(player_data, frame_shape)
            
            return player_data
            
        except Exception as e:
            return player_data
    
    def _select_best_two_players(self, player_data, frame_shape):
        """Select the best 2 players from multiple detections"""
        try:
            # Score players based on court position and confidence
            scored_players = []
            
            for player in player_data:
                score = 0
                
                # Higher confidence = better
                score += player['confidence'] * 10
                
                # Players in middle court area are more likely to be actual players
                court_pos = player.get('court_position', [0, 0])
                if 1 < court_pos[0] < 5 and 2 < court_pos[1] < 8:  # Middle court area
                    score += 5
                
                # Players with good pose detection are more likely to be actual players
                if player.get('pose') and player['pose'].pose_landmarks:
                    score += 3
                
                scored_players.append((score, player))
            
            # Sort by score and return top 2
            scored_players.sort(key=lambda x: x[0], reverse=True)
            return [player for score, player in scored_players[:2]]
            
        except Exception as e:
            return player_data[:2]
    
    def _find_missing_player(self, player_data, frame_shape):
        """Try to find the missing player when only 1 is detected"""
        try:
            # For now, just return the single player
            # In a full implementation, this could use tracking or prediction
            # to estimate where the second player might be
            
            # Simple heuristic: if we have one player, assume the other is in
            # the opposite court area
            if len(player_data) == 1:
                existing_player = player_data[0]
                court_pos = existing_player.get('court_position', [3.2, 4.875])  # Center court
                
                # Estimate second player position (opposite side)
                estimated_pos = [6.4 - court_pos[0], court_pos[1]]
                
                # Create a placeholder for the missing player
                # (This is a simplified approach - in practice, you'd use tracking)
                missing_player = {
                    'bbox': [0, 0, 0, 0],  # Placeholder bbox
                    'confidence': 0.3,  # Low confidence for estimated player
                    'pose': None,
                    'court_position': estimated_pos,
                    'label': 'Player 2 (estimated)'
                }
                
                return [existing_player, missing_player]
            
            return player_data
            
        except Exception as e:
            return player_data
    
    def _draw_annotations(self, frame, player_data, all_candidates=None):
        """Draw player and all person annotations on frame"""
        try:
            annotated_frame = frame.copy()
            # Draw all detected people (grey boxes)
            if all_candidates is not None:
                for candidate in all_candidates:
                    bbox = candidate['bbox']
                    confidence = candidate['confidence']
                    box_w = bbox[2] - bbox[0]
                    box_h = bbox[3] - bbox[1]
                    size_str = f"{box_w}x{box_h}"
                    # Draw thin grey box
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (128,128,128), 1, lineType=cv2.LINE_AA)
                    # Draw label with confidence and size
                    label = f"Person ({confidence:.2f}) {size_str}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 4), (bbox[0] + label_size[0], bbox[1]), (128,128,128), -1)
                    cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            # Draw player boxes (blue/green)
            for i, player in enumerate(player_data):
                bbox = player['bbox']
                confidence = player['confidence']
                box_w = bbox[2] - bbox[0]
                box_h = bbox[3] - bbox[1]
                size_str = f"{box_w}x{box_h}"
                pose = player['pose']
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                if len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0:
                    print(f"Debug: Skipping invalid bbox: {bbox}")
                    continue
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1, lineType=cv2.LINE_AA)
                label = f"{player['label']} ({confidence:.2f}) {size_str}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 6), (bbox[0] + label_size[0], bbox[1]), color, -1)
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
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
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1
            
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
                    
                    # Convert MediaPipe normalized coordinates to bbox-relative coordinates
                    start_x = int(bbox_x1 + start_landmark.x * bbox_w)
                    start_y = int(bbox_y1 + start_landmark.y * bbox_h)
                    end_x = int(bbox_x1 + end_landmark.x * bbox_w)
                    end_y = int(bbox_y1 + end_landmark.y * bbox_h)
                    
                    # Clamp coordinates to stay within frame bounds
                    start_x = max(0, min(frame_width - 1, start_x))
                    start_y = max(0, min(frame_height - 1, start_y))
                    end_x = max(0, min(frame_width - 1, end_x))
                    end_y = max(0, min(frame_height - 1, end_y))
                    
                    # Only draw if both points have reasonable visibility AND are within bbox
                    if (start_landmark.visibility > 0.1 and end_landmark.visibility > 0.1 and
                        bbox_x1 <= start_x <= bbox_x2 and bbox_y1 <= start_y <= bbox_y2 and
                        bbox_x1 <= end_x <= bbox_x2 and bbox_y1 <= end_y <= bbox_y2):
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