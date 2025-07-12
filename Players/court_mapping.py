import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

class SquashCourtMapper:
    """
    Maps player positions from camera view to top-down court view.
    
    Standard squash court dimensions (in meters):
    - Length: 9.75m
    - Width: 6.4m
    - Service line: 1.6m from back wall
    - Short line: 4.26m from back wall
    """
    
    def __init__(self, video_width=1920, video_height=1080):
        self.video_width = video_width
        self.video_height = video_height
        
        # Standard squash court dimensions (in meters)
        self.court_length = 9.75  # front to back
        self.court_width = 6.4   # side to side
        self.service_line = 1.6   # from back wall
        self.short_line = 4.26    # from back wall
        
        # Court visualization dimensions (pixels)
        self.court_display_width = 400
        self.court_display_height = int(400 * (self.court_length / self.court_width))
        
        # Default camera-to-court transformation matrix
        # This will need calibration for each specific camera angle
        self.transformation_matrix = self._create_default_transformation()
        
        # Player colors for visualization
        self.player_colors = ['red', 'blue']
        
    def _create_default_transformation(self):
        """
        Create a default perspective transformation matrix.
        This assumes a typical squash court camera angle.
        """
        # Source points (camera view) - these are typical for squash court cameras
        # Adjust these based on your specific camera setup
        src_points = np.float32([
            [0.15 * self.video_width, 0.3 * self.video_height],   # Top-left court corner
            [0.85 * self.video_width, 0.3 * self.video_height],   # Top-right court corner
            [0.05 * self.video_width, 0.95 * self.video_height],  # Bottom-left court corner
            [0.95 * self.video_width, 0.95 * self.video_height]   # Bottom-right court corner
        ])
        
        # Destination points (top-down court view)
        dst_points = np.float32([
            [0, 0],                                    # Top-left (front-left)
            [self.court_display_width, 0],            # Top-right (front-right)
            [0, self.court_display_height],           # Bottom-left (back-left)
            [self.court_display_width, self.court_display_height]  # Bottom-right (back-right)
        ])
        
        # Calculate perspective transformation matrix
        return cv2.getPerspectiveTransform(src_points, dst_points)
    
    def calibrate_court(self, frame, court_corners):
        """
        Calibrate the court mapping using manually selected court corners.
        
        Args:
            frame: Video frame for calibration
            court_corners: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                          in order: front-left, front-right, back-left, back-right
        """
        src_points = np.float32(court_corners)
        dst_points = np.float32([
            [0, 0],
            [self.court_display_width, 0],
            [0, self.court_display_height],
            [self.court_display_width, self.court_display_height]
        ])
        
        self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print("Court calibration completed!")
    
    def transform_player_position(self, player_bbox):
        """
        Transform player bounding box center to court coordinates.
        
        Args:
            player_bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (court_x, court_y) position on court
        """
        # Get player center point (feet position - bottom center of bbox)
        player_center_x = (player_bbox[0] + player_bbox[2]) / 2
        player_feet_y = player_bbox[3]  # Bottom of bounding box
        
        # Transform to court coordinates
        point = np.array([[[player_center_x, player_feet_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transformation_matrix)
        
        return transformed[0][0][0], transformed[0][0][1]
    
    def create_court_visualization(self):
        """Create a matplotlib figure with court layout."""
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
        
        # Main video view
        ax_video = fig.add_subplot(gs[0])
        ax_video.set_title('Video Feed with Player Detection')
        ax_video.set_aspect('equal')
        
        # Court map view
        ax_court = fig.add_subplot(gs[1])
        ax_court.set_title('Court Map (Top View)')
        ax_court.set_xlim(0, self.court_display_width)
        ax_court.set_ylim(0, self.court_display_height)
        ax_court.invert_yaxis()  # Invert Y-axis so front of court is at top
        ax_court.set_aspect('equal')
        
        # Draw court lines
        self._draw_court_lines(ax_court)
        
        return fig, ax_video, ax_court
    
    def _draw_court_lines(self, ax):
        """Draw squash court lines on the court map."""
        # Court boundaries
        court_rect = patches.Rectangle((0, 0), self.court_display_width, 
                                     self.court_display_height, 
                                     linewidth=2, edgecolor='black', facecolor='lightgray')
        ax.add_patch(court_rect)
        
        # Service line (1.6m from back wall)
        service_line_y = self.court_display_height * (1 - self.service_line / self.court_length)
        ax.axhline(y=service_line_y, color='black', linewidth=1, linestyle='--')
        ax.text(self.court_display_width/2, service_line_y-10, 'Service Line', 
                ha='center', va='top', fontsize=8)
        
        # Short line (4.26m from back wall)
        short_line_y = self.court_display_height * (1 - self.short_line / self.court_length)
        ax.axhline(y=short_line_y, color='black', linewidth=1, linestyle='--')
        ax.text(self.court_display_width/2, short_line_y-10, 'Short Line', 
                ha='center', va='top', fontsize=8)
        
        # Center line
        ax.axvline(x=self.court_display_width/2, color='black', linewidth=1, linestyle='--')
        
        # Service boxes
        service_box_width = self.court_display_width / 4
        left_service = patches.Rectangle((service_box_width, service_line_y), 
                                       service_box_width, 
                                       short_line_y - service_line_y,
                                       linewidth=1, edgecolor='black', facecolor='none')
        right_service = patches.Rectangle((2*service_box_width, service_line_y), 
                                        service_box_width, 
                                        short_line_y - service_line_y,
                                        linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(left_service)
        ax.add_patch(right_service)
        
        # Labels
        ax.text(self.court_display_width/2, 10, 'FRONT WALL', 
                ha='center', va='bottom', fontsize=10, weight='bold')
        ax.text(self.court_display_width/2, self.court_display_height-10, 'BACK WALL', 
                ha='center', va='top', fontsize=10, weight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    def update_player_positions(self, ax_court, player_positions):
        """
        Update player positions on the court map.
        
        Args:
            ax_court: Court matplotlib axis
            player_positions: List of (x, y) positions on court
        """
        # Clear previous player markers
        for artist in ax_court.get_children():
            if isinstance(artist, plt.Circle):
                artist.remove()
        
        # Draw new player positions
        for i, (x, y) in enumerate(player_positions):
            if x is not None and y is not None:
                # Ensure position is within court bounds
                x = max(0, min(x, self.court_display_width))
                y = max(0, min(y, self.court_display_height))
                
                circle = plt.Circle((x, y), 8, color=self.player_colors[i], 
                                  alpha=0.8, zorder=10)
                ax_court.add_patch(circle)
                
                # Add player label
                ax_court.text(x, y-15, f'P{i+1}', ha='center', va='top', 
                            fontsize=8, weight='bold', color=self.player_colors[i])


def create_court_mapping_demo():
    """
    Create a demonstration of the court mapping system.
    This can be used to test and visualize the mapping.
    """
    # Create court mapper
    mapper = SquashCourtMapper()
    
    # Create visualization
    fig, ax_video, ax_court = mapper.create_court_visualization()
    
    # Demo: simulate player positions
    demo_positions = [
        [(100, 300), (300, 200)],  # Frame 1
        [(120, 280), (280, 220)],  # Frame 2
        [(140, 260), (260, 240)],  # Frame 3
        [(160, 240), (240, 260)],  # Frame 4
        [(180, 220), (220, 280)],  # Frame 5
    ]
    
    def animate(frame):
        if frame < len(demo_positions):
            mapper.update_player_positions(ax_court, demo_positions[frame])
        return []
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(demo_positions), 
                        interval=1000, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return mapper, fig, anim


if __name__ == "__main__":
    # Run demo
    mapper, fig, anim = create_court_mapping_demo()
    print("Court mapping demo created!")
    print("The court map shows player positions as colored dots.")
    print("Red = Player 1, Blue = Player 2")