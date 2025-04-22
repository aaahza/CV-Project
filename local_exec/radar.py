import argparse
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Constants
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football_detector/weights/best.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/field_detector3/weights/best.pt')

GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

# Annotators
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list:
    """Extract crops from the frame based on detected bounding boxes."""
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """Resolve team IDs for goalkeepers based on proximity to team centroids."""
    if len(goalkeepers) == 0:
        return np.array([])

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Handle case where there might not be any players from a team
    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]
    
    team_0_centroid = team_0_players.mean(axis=0) if len(team_0_players) > 0 else np.array([0, 0])
    team_1_centroid = team_1_players.mean(axis=0) if len(team_1_players) > 0 else np.array([0, 0])
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    """Render a radar view of players on the pitch."""
    # Filter valid keypoints
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not any(mask) or len(keypoints.xy[0][mask]) < 4:
        # Return an empty radar if not enough keypoints
        pitch = draw_pitch(config=CONFIG)
        return pitch
    
    # Create view transformer
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    
    # Transform player positions to pitch coordinates
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # Draw the radar
    radar = draw_pitch(config=CONFIG)
    
    # Draw each team's players
    for team_id in range(4):  # 0, 1: teams, 2: referee, 3: other officials
        team_positions = transformed_xy[color_lookup == team_id]
        if len(team_positions) > 0:
            radar = draw_points_on_pitch(
                config=CONFIG, xy=team_positions,
                face_color=sv.Color.from_hex(COLORS[team_id]), 
                radius=20, pitch=radar
            )
    
    return radar


def run_radar(source_video_path: str, device: str = "cpu", show_display: bool = False):
    """
    Run the radar visualization and return annotated frames.
    
    Args:
        source_video_path: Path to the source video
        device: Device to use for inference ('cpu' or 'cuda')
        show_display: Whether to show frames in a window while processing
        
    Returns:
        Generator yielding annotated frames
    """
    # Load models
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    
    # First pass: collect player crops for team classification
    print("Collecting player samples for team classification...")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='Collecting player crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    # Train team classifier on collected crops
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Second pass: actual radar processing
    print("Processing video with radar visualization...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in frame_generator:
        # Detect pitch keypoints
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        
        # Detect players
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        # Separate players by class
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Classify teams
        player_crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(player_crops) if player_crops else np.array([])
        
        # Assign goalkeepers to teams
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        # Merge detections and create color lookup
        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        
        # Create labels (tracker IDs)
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        # Annotate players in the frame
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        # Create and overlay the radar
        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        
        if show_display:
            cv2.imshow("Radar View", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        yield annotated_frame


def main():
    """Main function to run the radar."""

    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Football Radar Visualization')
    parser.add_argument('--source_video_path', type=str, required=True,
                        help='Path to the source video')
    parser.add_argument('--target_video_path', type=str, required=True,
                        help='Path to save the output video')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (cpu or cuda)')
    args = parser.parse_args()

    # Process video and generate output
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in run_radar(
            source_video_path=args.source_video_path,
            device=args.device,
            show_display=False
        ):
            sink.write_frame(frame)
    
    # cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved to {args.target_video_path}")
    end_time = time.time()
    print(f"Video processing complete. Output saved to {args.target_video_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()