import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from player_detection_lib import detect_players, assign_player_labels, load_yolo_model
from court_mapping import SquashCourtMapper
import threading
import time
from collections import deque

class PlayerCourtTracker:
    """
    Real-time player tracking with court mapping visualization.
    Combines YOLO player detection with court position mapping.
    """
    
    def __init__(self, video_path, model_path='yolov8n.pt', crop_margin_percent=0.10):
        self.video_path = video_path
        self.crop_margin_percent = crop_margin_percent
        
        # Load YOLO model
        self.model = load_yolo_model(model_path)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate crop dimensions
        self.crop_margin = int(self.crop_margin_percent * self.width)
        self.cropped_width = self.width - 2 * self.crop_margin
        
        # Initialize court mapper
        self.court_mapper = SquashCourtMapper(self.cropped_width, self.height)
        
        # Player tracking
        self.prev_centroids = None
        self.player_positions_history = deque(maxlen=100)  # Store last 100 positions
        
        # Visualization
        self.fig = None
        self.ax_video = None
        self.ax_court = None
        self.video_img = None
        self.court_trails = [deque(maxlen=20), deque(maxlen=20)]  # Player trails
        
        # Control flags
        self.running = False
        self.paused = False
        
    def setup_visualization(self):
        """Set up the matplotlib visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1], figure=self.fig)
        
        # Main video view
        self.ax_video = self.fig.add_subplot(gs[0, 0])
        self.ax_video.set_title('Live Video Feed with Player Detection')
        self.ax_video.set_xticks([])
        self.ax_video.set_yticks([])
        
        # Court map view
        self.ax_court = self.fig.add_subplot(gs[0, 1])
        self.ax_court.set_title('Court Map (Top View)')
        self.ax_court.set_xlim(0, self.court_mapper.court_display_width)
        self.ax_court.set_ylim(0, self.court_mapper.court_display_height)
        self.ax_court.invert_yaxis()
        self.ax_court.set_aspect('equal')
        
        # Draw court lines
        self.court_mapper._draw_court_lines(self.ax_court)
        
        # Player statistics view
        self.ax_stats = self.fig.add_subplot(gs[1, :])
        self.ax_stats.set_title('Player Movement Statistics')
        self.ax_stats.set_xlabel('Time (seconds)')
        self.ax_stats.set_ylabel('Court Position')\n        
        # Initialize video display
        dummy_frame = np.zeros((self.height, self.cropped_width, 3), dtype=np.uint8)
        self.video_img = self.ax_video.imshow(dummy_frame)
        
        plt.tight_layout()
        return self.fig
    
    def calibrate_court_interactive(self):
        """
        Interactive court calibration.
        Click on the four corners of the court in the video.
        """
        print("Court Calibration Mode")
        print("Click on the four corners of the court in this order:")
        print("1. Front-left corner")
        print("2. Front-right corner") 
        print("3. Back-left corner")
        print("4. Back-right corner")
        print("Press 'q' when done or 'r' to reset")
        
        # Get a frame for calibration
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame for calibration")
            return False
        
        # Crop frame
        cropped_frame = frame[:, self.crop_margin:self.width-self.crop_margin]
        
        # Store clicked points
        clicked_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_points.append((x, y))
                print(f"Point {len(clicked_points)}: ({x}, {y})")
                
                # Draw point on frame
                cv2.circle(cropped_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(cropped_frame, str(len(clicked_points)), 
                           (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(clicked_points) == 4:
                    print("All points selected. Press 'q' to confirm calibration.")
        
        # Set up display window
        cv2.namedWindow('Court Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Court Calibration', mouse_callback)
        
        while True:
            display_frame = cropped_frame.copy()
            
            # Draw instructions
            cv2.putText(display_frame, f"Click corner {len(clicked_points)+1}/4", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Court Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if len(clicked_points) == 4:
                    # Perform calibration
                    self.court_mapper.calibrate_court(cropped_frame, clicked_points)
                    print("Calibration completed!")
                    break
                else:
                    print(f"Need 4 points, only have {len(clicked_points)}")
            elif key == ord('r'):
                # Reset points
                clicked_points = []
                cropped_frame = frame[:, self.crop_margin:self.width-self.crop_margin].copy()
                print("Points reset")
        
        cv2.destroyWindow('Court Calibration')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to beginning
        return len(clicked_points) == 4
    
    def process_frame(self, frame):
        """Process a single frame for player detection and court mapping."""
        # Crop frame
        cropped_frame = frame[:, self.crop_margin:self.width-self.crop_margin]
        
        # Detect players
        player_boxes, centroids = detect_players(self.model, cropped_frame)
        
        # Assign player labels
        labels, self.prev_centroids = assign_player_labels(centroids, self.prev_centroids)
        
        # Transform to court coordinates
        court_positions = []
        for bbox in player_boxes:
            if len(bbox) == 4:
                court_x, court_y = self.court_mapper.transform_player_position(bbox)
                court_positions.append((court_x, court_y))
            else:
                court_positions.append((None, None))
        
        # Ensure we have 2 positions (pad with None if needed)
        while len(court_positions) < 2:
            court_positions.append((None, None))
        
        # Store positions in history
        self.player_positions_history.append(court_positions[:2])
        
        # Update trails
        for i in range(2):
            if court_positions[i][0] is not None:
                self.court_trails[i].append(court_positions[i])
        
        # Draw annotations on cropped frame
        annotated_frame = cropped_frame.copy()
        if len(player_boxes) == 2:
            for (x1, y1, x2, y2), label in zip(player_boxes, labels):
                label_str = label if label else "Player"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label_str, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return annotated_frame, court_positions[:2]
    
    def update_visualization(self, frame, court_positions):
        """Update the visualization with new frame and positions."""
        # Update video display
        self.video_img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Clear previous court markers
        for artist in self.ax_court.get_children():
            if isinstance(artist, plt.Circle) or isinstance(artist, plt.Line2D):
                artist.remove()
        
        # Draw player trails
        for i, trail in enumerate(self.court_trails):
            if len(trail) > 1:
                trail_x = [pos[0] for pos in trail if pos[0] is not None]
                trail_y = [pos[1] for pos in trail if pos[1] is not None]
                if len(trail_x) > 1:
                    self.ax_court.plot(trail_x, trail_y, 
                                     color=self.court_mapper.player_colors[i], 
                                     alpha=0.5, linewidth=2)
        
        # Draw current player positions
        for i, (x, y) in enumerate(court_positions):
            if x is not None and y is not None:
                # Ensure position is within court bounds
                x = max(0, min(x, self.court_mapper.court_display_width))
                y = max(0, min(y, self.court_mapper.court_display_height))
                
                circle = plt.Circle((x, y), 10, color=self.court_mapper.player_colors[i], 
                                  alpha=0.8, zorder=10)
                self.ax_court.add_patch(circle)
                
                # Add player label
                self.ax_court.text(x, y-20, f'P{i+1}', ha='center', va='top', 
                                 fontsize=10, weight='bold', 
                                 color=self.court_mapper.player_colors[i])
        
        # Redraw court lines to ensure they're on top
        self.court_mapper._draw_court_lines(self.ax_court)
        
        plt.pause(0.001)  # Small pause to allow GUI updates
    
    def run_tracking(self):
        """Run the real-time player tracking."""
        self.running = True
        frame_count = 0
        
        print("Starting player tracking...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        while self.running:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Process frame
                annotated_frame, court_positions = self.process_frame(frame)
                
                # Update visualization
                self.update_visualization(annotated_frame, court_positions)
                
                frame_count += 1
                
                # Control playback speed
                time.sleep(1.0 / self.fps)
            else:
                time.sleep(0.1)  # Small sleep when paused
        
        print(f"Tracking completed. Processed {frame_count} frames.")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        plt.close('all')


def main():
    """Main function to run the player court tracker."""
    import os
    
    # Get video file
    video_dir = "annotated_main_camera_segments"
    if not os.path.exists(video_dir):
        print(f"Directory {video_dir} not found. Please run player detection first.")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    # Use first video file
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Using video: {video_files[0]}")
    
    # Create tracker
    tracker = PlayerCourtTracker(video_path)
    
    # Set up visualization
    fig = tracker.setup_visualization()
    
    # Optional: Run calibration
    print("\nDo you want to calibrate the court mapping? (y/n)")
    if input().lower() == 'y':
        if not tracker.calibrate_court_interactive():
            print("Calibration failed, using default mapping")
    
    # Run tracking
    try:
        tracker.run_tracking()
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    finally:
        tracker.cleanup()


if __name__ == "__main__":
    main()