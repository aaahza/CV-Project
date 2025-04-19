import supervision as sv
import cv2
import numpy as np
from .graphics import draw_team_ball_control

# Enhanced color scheme with more professional colors
colors = {
    "team1": sv.ColorPalette.from_hex(['#1E90FF']),  # Blue
    "team2": sv.ColorPalette.from_hex(['#DC143C']),  # Red
    "referee": sv.ColorPalette.from_hex(['#FFD700']),  # Gold
    "goalkepper": sv.ColorPalette.from_hex(['#00FF00']),  # Green (more visible)
    "label_text": sv.Color.from_hex('#FFFFFF'),  # White text
    "label_background": sv.Color.from_hex('#000000'),  # Black background
    "ball": sv.Color.from_hex('#FF8C00'),  # Orange
    "active_player": sv.Color.from_rgb_tuple((255, 0, 0)),  # Bright red
    "scoreboard_bg": sv.Color.from_rgb_tuple((0, 0, 0)),  # Black (we'll handle transparency separately)
    "scoreboard_text": sv.Color.from_hex('#FFFFFF'),  # White
    "heatmap": sv.Color.from_hex('#FF0000')  # Red for heatmap
}

LABEL_TEXT_POSITION = sv.Position.BOTTOM_CENTER

# Enhanced annotators with better visibility
team1_ellipse_annotator = sv.EllipseAnnotator(color=colors["team1"], thickness=2)  
team2_ellipse_annotator = sv.EllipseAnnotator(color=colors['team2'], thickness=2)
referee_ellipse_annotator = sv.EllipseAnnotator(color=colors['referee'], thickness=2)
goalkepper_ellipse_annotator = sv.EllipseAnnotator(color=colors['goalkepper'], thickness=2)
active_player_annotator = sv.TriangleAnnotator(
    color=colors['active_player'],
    base=20,
    height=20,
    outline_color=sv.Color.BLACK,
    outline_thickness=2
)

# Enhanced label annotators with better readability - using only supported parameters
team1_label_annotator = sv.LabelAnnotator(
    color=colors['team1'],
    text_color=colors['label_text'],
    text_position=LABEL_TEXT_POSITION,
    text_scale=0.6,
    text_thickness=1
)
team2_label_annotator = sv.LabelAnnotator(
    color=colors['team2'],
    text_color=colors['label_text'],
    text_position=LABEL_TEXT_POSITION,
    text_scale=0.6,
    text_thickness=1
)
referee_label_annotator = sv.LabelAnnotator(
    color=colors['referee'],
    text_color=colors['label_text'],
    text_position=LABEL_TEXT_POSITION,
    text_scale=0.6,
    text_thickness=1
)
goalkepper_label_annotator = sv.LabelAnnotator(
    color=colors['goalkepper'],
    text_color=colors['label_text'],
    text_position=LABEL_TEXT_POSITION,
    text_scale=0.6,
    text_thickness=1
)

ball_triangle_annotator = sv.TriangleAnnotator(
    color=colors['ball'],
    base=20,
    height=20,
    outline_color=sv.Color.BLACK,
    outline_thickness=2
)

# Track positions for heatmap generation
team1_positions = []
team2_positions = []
max_positions = 200  # Keep the last 200 positions

# Function to add scoreboard overlay
def add_scoreboard(frame, ball_possession):
    height, width = frame.shape[:2]
    
    # Calculate possession percentages
    total_possession = sum(ball_possession.values())
    team1_pct = int(ball_possession[list(ball_possession.keys())[0]] / max(total_possession, 1) * 100)
    team2_pct = int(ball_possession[list(ball_possession.keys())[1]] / max(total_possession, 1) * 100)
    
    # Create scoreboard background
    overlay = frame.copy()
    
    # Top scoreboard
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    
    # Add text for possession stats
    cv2.putText(
        overlay, 
        f"Team 1: {team1_pct}%", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (29, 144, 255), 
        2
    )
    
    cv2.putText(
        overlay, 
        f"BALL POSSESSION", 
        (width//2 - 80, 25), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        1
    )
    
    cv2.putText(
        overlay, 
        f"Team 2: {team2_pct}%", 
        (width - 150, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (220, 20, 60), 
        2
    )
    
    # Blend the scoreboard with the original frame
    alpha = 0.8
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame

# Generate heatmap from positions
def generate_heatmap(frame, positions, color):
    if not positions:
        return frame
        
    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.uint8)
    
    # Draw points in the heatmap
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(heatmap, (x, y), 15, 255, -1)
    
    # Blur the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original frame
    alpha = 0.3
    result = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
    
    return result

# Enhanced frame annotation with additional visualizations
def annotate_frames(frame, all_detection, labels, ball_possession, show_heatmap=False):
    annotated_frame = frame.copy()
    
    # Track player positions for heatmap
    if len(all_detection["team1"].xyxy) > 0:
        for box in all_detection["team1"].xyxy:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            team1_positions.append((center_x, center_y))
    
    if len(all_detection["team2"].xyxy) > 0:
        for box in all_detection["team2"].xyxy:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            team2_positions.append((center_x, center_y))
    
    # Limit the number of positions stored
    if len(team1_positions) > max_positions:
        team1_positions.pop(0)
    if len(team2_positions) > max_positions:
        team2_positions.pop(0)
    
    # Optional heatmap visualization
    if show_heatmap:
        team1_heatmap = generate_heatmap(annotated_frame, team1_positions, colors["team1"])
        team2_heatmap = generate_heatmap(team1_heatmap, team2_positions, colors["team2"])
        annotated_frame = team2_heatmap
    
    # Standard annotations
    annotated_frame = team1_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["team1"])
    annotated_frame = team2_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["team2"])
    annotated_frame = referee_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["referee"])
    annotated_frame = goalkepper_ellipse_annotator.annotate(scene=annotated_frame, detections=all_detection["goalkeepers"])
    annotated_frame = ball_triangle_annotator.annotate(scene=annotated_frame, detections=all_detection["ball"])
    
    # Add labels
    annotated_frame = team1_label_annotator.annotate(
        scene=annotated_frame, 
        detections=all_detection["team1"], 
        labels=labels["labels_team1"]
    )
    annotated_frame = team2_label_annotator.annotate(
        scene=annotated_frame, 
        detections=all_detection["team2"], 
        labels=labels["labels_team2"]
    )
    annotated_frame = referee_label_annotator.annotate(
        scene=annotated_frame,
        detections=all_detection["referee"],
        labels=labels["labels_referee"]
    )
    annotated_frame = goalkepper_label_annotator.annotate(
        scene=annotated_frame,
        detections=all_detection["goalkeepers"],
        labels=labels["labels_gk"]
    )
    
    # Highlight active player
    annotated_frame = active_player_annotator.annotate(
        scene=annotated_frame,
        detections=all_detection["active_player"]
    )
    
    # Add the scoreboard
    annotated_frame = add_scoreboard(annotated_frame, ball_possession)
    
    return annotated_frame