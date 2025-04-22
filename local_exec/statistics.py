import argparse
import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from tqdm import tqdm

# Import necessary modules from sports package
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch

# Constants
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/football_detector/weights/best.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'models/field_detector3/weights/best.pt')

# Soccer field configuration
CONFIG = SoccerPitchConfiguration()

# Real-world measurements for soccer pitch (in meters)
PITCH_LENGTH = 105  # standard soccer pitch length
PITCH_WIDTH = 68    # standard soccer pitch width

# Colors and annotators
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
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
SPEED_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.TOP_CENTER,
)
DISTANCE_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_RIGHT,
)


class SpeedTracker:
    """Tracks player positions and calculates speeds in real-world units (m/s) and accumulated distance (m)"""
    
    def __init__(self, frame_window=5, fps=30):
        """
        Initialize the speed tracker.
        
        Args:
            frame_window: Number of frames to use for speed calculation window
            fps: Frames per second of the video
        """
        self.positions = defaultdict(list)
        self.speeds = defaultdict(float)
        self.total_distance = defaultdict(float)
        self.frame_window = frame_window
        self.fps = fps
        # Scale factor to convert from cm to meters
        self.scale_factor = 0.01
        
    def update(self, detections, transformed_xy, tracker_ids):
        """
        Update player positions and calculate speeds.
        
        Args:
            detections: Supervision Detections object
            transformed_xy: Transformed coordinates on 2D pitch (in cm)
            tracker_ids: IDs of tracked objects
            
        Returns:
            Tuple of dictionaries mapping tracker IDs to speeds (m/s) and distances (m)
        """
        if len(tracker_ids) == 0:
            return {}, {}
            
        # Update positions for each tracked player
        for i, tracker_id in enumerate(tracker_ids):
            if i < len(transformed_xy):
                # Store position in the 2D pitch coordinate system
                position = transformed_xy[i]
                
                # Add to position history
                if tracker_id in self.positions:
                    # Calculate distance from last position
                    last_pos = self.positions[tracker_id][-1]
                    displacement = np.linalg.norm(position - last_pos)
                    
                    # Convert to meters and add to total distance
                    displacement_meters = displacement * self.scale_factor
                    self.total_distance[tracker_id] += displacement_meters
                    
                self.positions[tracker_id].append(position)
                
                # Calculate speed only if we have enough positions
                if len(self.positions[tracker_id]) >= self.frame_window:
                    # Get positions at the start and end of the window
                    start_position = self.positions[tracker_id][-self.frame_window]
                    end_position = self.positions[tracker_id][-1]
                    
                    # Calculate displacement in cm
                    displacement = np.linalg.norm(end_position - start_position)
                    
                    # Convert to meters
                    displacement_meters = displacement * self.scale_factor
                    
                    # Calculate time elapsed in seconds
                    time_elapsed = (self.frame_window - 1) / self.fps
                    
                    # Calculate speed in m/s
                    if time_elapsed > 0:
                        speed_m_s = displacement_meters / time_elapsed
                        
                        # Apply smoothing (70% previous value, 30% new value)
                        if tracker_id in self.speeds:
                            speed_m_s = 0.7 * self.speeds[tracker_id] + 0.3 * speed_m_s
                            
                        # Store the speed
                        self.speeds[tracker_id] = speed_m_s
                
                # Trim history to save memory (keep 2x window size)
                if len(self.positions[tracker_id]) > 2 * self.frame_window:
                    self.positions[tracker_id] = self.positions[tracker_id][-2 * self.frame_window:]
        
        return self.speeds, self.total_distance


def run_player_speed_tracking(source_video_path: str, device: str = "cpu"):
    """
    Run player tracking with speed calculation and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with player tracking and speeds.
    """
    # Load models
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    
    # Get video info for fps calculation
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps
    
    # Initialize tracker and speed tracker
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    speed_tracker = SpeedTracker(frame_window=5, fps=fps)
    
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # Show progress
    print("Processing frames for player speed tracking...")
    
    for frame in frame_generator:
        # Detect pitch keypoints
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        
        # Detect and track players
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        
        # Labels for player IDs
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        
        # Create annotated frame
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        
        # Get player positions in image space
        if len(detections) > 0:
            # Create homography transformation if we have valid keypoints
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if any(mask) and len(keypoints.xy[0][mask]) >= 4:
                # Create view transformer
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                )
                
                # Transform player positions to pitch coordinates
                xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_xy = transformer.transform_points(points=xy)
                
                # Calculate speeds and distances
                speeds, distances = speed_tracker.update(detections, transformed_xy, detections.tracker_id)
                
                # Create labels for speed and distance with both m/s and km/h
                speed_labels = [f"{speeds.get(tid, 0):.1f} m/s ({speeds.get(tid, 0) * 3.6:.1f} km/h)" for tid in detections.tracker_id]
                annotated_frame = SPEED_ANNOTATOR.annotate(
                    annotated_frame, detections, labels=speed_labels)
                
                distance_labels = [f"{distances.get(tid, 0):.1f} m" for tid in detections.tracker_id]
                annotated_frame = DISTANCE_ANNOTATOR.annotate(
                    annotated_frame, detections, labels=distance_labels)
                
                # Optional: Render the pitch transform as a small overlay
                h, w, _ = frame.shape
                pitch = draw_pitch(config=CONFIG)
                pitch = cv2.resize(pitch, (w // 4, h // 4))
                
                # Draw player positions on pitch
                for i, pos in enumerate(transformed_xy):
                    # Scale position to fit the small pitch visualization
                    x = int(pos[0] * w / (4 * CONFIG.width))
                    y = int(pos[1] * h / (4 * CONFIG.length))
                    
                    # Make sure coordinates are within bounds
                    if 0 <= x < w//4 and 0 <= y < h//4:
                        cv2.circle(pitch, (x, y), 3, (0, 0, 255), -1)
                
                # Add mini-map to top-right corner
                annotated_frame[0:h//4, w-w//4:w] = pitch
        
        yield annotated_frame


def process_video_with_speed(source_video_path: str, target_video_path: str, device: str = "cpu"):
    """
    Process a video file with player tracking, speed calculation and save the result.

    Args:
        source_video_path (str): Path to the source video.
        target_video_path (str): Path to save the processed video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
    """
    # Get frames with player tracking and speed calculations
    frame_generator = run_player_speed_tracking(
        source_video_path=source_video_path, device=device)

    # Setup video writer
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    
    print(f"Processing video: {source_video_path}")
    print(f"Output will be saved to: {target_video_path}")
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            # Display the frame (optional)
        #     cv2.imshow("Player Speed Tracking", frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        # cv2.destroyAllWindows()
    
    print("Video processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track players and measure speeds in soccer videos')
    parser.add_argument('--source_video_path', type=str, required=True, 
                        help='Path to the source video')
    parser.add_argument('--target_video_path', type=str, required=True, 
                        help='Path to save the output video')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run on (cpu, cuda, etc.)')
    parser.add_argument('--mode', type=str, default='speed',
                        choices=['basic', 'speed'],
                        help='Tracking mode: basic or with speed calculations')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        # Use the original player tracking function
        process_video(
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            device=args.device
        )
    else:
        # Use the new player speed tracking function
        process_video_with_speed(
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            device=args.device
        )