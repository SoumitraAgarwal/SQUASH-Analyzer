import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def load_yolo_model(model_path='yolov8n.pt'):
    """Load YOLO model for player detection."""
    return YOLO(model_path)

def detect_players(model, frame):
    """
    Detect players in a frame using YOLO model.
    
    Args:
        model: YOLO model instance
        frame: Input frame
        
    Returns:
        List of player bounding boxes [(x1, y1, x2, y2), ...]
        List of centroids [(cx, cy), ...]
    """
    results = model(frame, verbose=False)
    player_boxes = []
    centroids = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 0 is 'person' in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                player_boxes.append((x1, y1, x2, y2))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroids.append((cx, cy))
    
    return player_boxes, centroids

def assign_player_labels(centroids, prev_centroids):
    """
    Assign consistent player labels based on position tracking.
    
    Args:
        centroids: Current frame centroids [(cx, cy), ...]
        prev_centroids: Previous frame centroids [(cx, cy), ...]
        
    Returns:
        List of labels ["Player 1", "Player 2", ...]
        Updated prev_centroids for next frame
    """
    labels = [f"Player {i+1}" for i in range(len(centroids))]
    
    if len(centroids) == 2:
        # If previous centroids exist, match by minimum distance
        if prev_centroids is not None and len(prev_centroids) == 2:
            dists = np.zeros((2, 2))
            for i, c in enumerate(centroids):
                for j, pc in enumerate(prev_centroids):
                    dists[i, j] = np.hypot(c[0] - pc[0], c[1] - pc[1])
            
            # Optimal assignment for 2 players
            if dists[0, 0] + dists[1, 1] <= dists[0, 1] + dists[1, 0]:
                labels = ["Player 1", "Player 2"]
                prev_centroids = [centroids[0], centroids[1]]
            else:
                labels = ["Player 2", "Player 1"]
                prev_centroids = [centroids[1], centroids[0]]
        else:
            # Assign leftmost as Player 1, rightmost as Player 2
            idxs = np.argsort([c[0] for c in centroids])
            labels = [""] * len(centroids)
            labels[idxs[0]] = "Player 1"
            labels[idxs[1]] = "Player 2"
            prev_centroids = [centroids[idxs[0]], centroids[idxs[1]]]
    elif len(centroids) == 1:
        # Single player detected
        labels = ["Player 1"]
        prev_centroids = [centroids[0]]
    elif len(centroids) == 0:
        # No players detected
        labels = []
        # Keep previous centroids for next frame
    
    return labels, prev_centroids

def draw_player_annotations(frame, player_boxes, labels):
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input frame
        player_boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        labels: List of player labels ["Player 1", "Player 2", ...]
        
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    
    if len(player_boxes) == 2:  # Only annotate if exactly 2 players detected
        for (x1, y1, x2, y2), label in zip(player_boxes, labels):
            label_str = label if label else "Player"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label_str, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated_frame

def process_video_with_player_detection(input_path, output_path, model_path='yolov8n.pt', 
                                      crop_margin_percent=0.10, show_progress=True, 
                                      inference_stride=3, progress_position=0):
    """
    Process a video file with player detection and annotation.
    
    Args:
        input_path: Path to input video
        output_path: Path to save annotated video
        model_path: Path to YOLO model
        crop_margin_percent: Percentage to crop from sides (0.10 = 10%)
        show_progress: Whether to show progress bar
        inference_stride: Run inference every N frames (3 = every 3rd frame)
        progress_position: Position for nested progress bars (0=top, 1=below, etc.)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load YOLO model
        model = load_yolo_model(model_path)
        
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
        crop_margin = int(crop_margin_percent * width)
        cropped_width = width - 2 * crop_margin
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (cropped_width, height))
        
        # Initialize player tracking
        prev_centroids = None
        last_player_boxes = []
        last_labels = []
        
        # Progress tracking
        progress_iter = None
        if show_progress:
            progress_iter = tqdm(
                total=total_frames, 
                desc=f"Processing {os.path.basename(input_path)[:50]}", 
                position=progress_position,
                leave=False,  # Clean up progress bar when done
                unit='frames',
                unit_scale=True
            )
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame
            cropped_frame = frame[:, crop_margin:width-crop_margin]
            
            # Run inference only on certain frames for speed
            if frame_idx % inference_stride == 0:
                # Detect players
                player_boxes, centroids = detect_players(model, cropped_frame)
                
                # Assign player labels
                labels, prev_centroids = assign_player_labels(centroids, prev_centroids)
                
                # Cache results for interpolation
                last_player_boxes = player_boxes
                last_labels = labels
            else:
                # Use cached results for non-inference frames
                player_boxes = last_player_boxes
                labels = last_labels
            
            # Draw annotations
            annotated_frame = draw_player_annotations(cropped_frame, player_boxes, labels)
            
            # Write frame
            out.write(annotated_frame)
            
            if progress_iter:
                progress_iter.update(1)
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        out.release()
        if progress_iter:
            progress_iter.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing video {input_path}: {str(e)}")
        return False