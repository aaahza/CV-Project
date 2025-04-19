from ultralytics import YOLO
import supervision as sv
import cv2
import argparse
from tqdm import tqdm
from utils import get_number_of_frames, annotate_frames, assign_ball_to_player, get_frames
from team_assigner import Assigner
import numpy as np
import os
from config import *


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Football Analysis with Computer Vision')
    parser.add_argument('--heatmap', action='store_true', help='Enable heatmap visualization')
    parser.add_argument('--input', type=str, default=VIDEO_SRC, help='Path to input video')
    parser.add_argument('--output', type=str, default=OUT_VIDEO, help='Path to output video')
    parser.add_argument('--model', type=str, default=MODEL_SRC, help='Path to YOLO model')
    parser.add_argument('--skip', type=int, default=0, help='Process every Nth frame (1=every frame, 2=every second frame, etc)')
    parser.add_argument('--optimize', action='store_true', help='Enable performance optimizations')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for resizing frames before processing (0.5 = half resolution)')
    args = parser.parse_args()
    
    # Use arguments or defaults
    video_src = args.input
    output_path = args.output
    model_src = args.model
    show_heatmap = args.heatmap
    skip_frames = args.skip
    optimize = args.optimize
    scale_factor = args.scale
    
    print(f"Loading model: {model_src}")
    model = YOLO(model_src)
    
    # get the total frames and FPS
    print(f"Analyzing video: {video_src}")
    total_frames, fps = get_number_of_frames(video_src)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # get the frames
    frame_generator = get_frames(video_src)

    # Get the first frame to determine the video dimensions
    first_frame = next(frame_generator)
    original_height, original_width, _ = first_frame.shape
    print(f"Video dimensions: {original_width}x{original_height}")
    
    # Calculate new dimensions if scaling is applied
    if scale_factor != 1.0:
        width = int(original_width * scale_factor)
        height = int(original_height * scale_factor)
        print(f"Processing at scaled dimensions: {width}x{height}")
    else:
        width, height = original_width, original_height

    # Initialize the video writer
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For Windows, try mp4v directly since we know avc1 fails
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))
    
    if not out.isOpened():
        raise Exception("Failed to create video writer. Check that the output directory exists and is writable.")
    
    # Initialize the ByteTrack object
    tracker = sv.ByteTrack()
    tracker.reset()
    tracker1 = sv.ByteTrack()
    tracker1.reset()

    # Process each frame in the video
    team_colors = {}
    is_first_frame = True
    class_id_active_player = None
    kmeans = None
    ball_posession = {MODEL_CLASSES["team1"]: 0, MODEL_CLASSES["team2"]: 0}
    
    # Reset frame generator to include the first frame
    frame_generator = get_frames(video_src)
    
    # Stats tracking
    frames_processed = 0
    frames_skipped = 0
    detections_count = {
        "ball": 0,
        "team1": 0,
        "team2": 0,
        "referee": 0,
        "goalkeeper": 0
    }
    
    # Set YOLO model parameters for optimization
    if optimize:
        print("Performance optimizations enabled")
        model.overrides['conf'] = 0.25  # Lower confidence threshold for speed
        model.overrides['iou'] = 0.6    # Higher IoU for faster NMS
    
    # Determine frame skipping
    process_every_n_frames = max(1, skip_frames) if skip_frames > 0 else 1
    if process_every_n_frames > 1:
        print(f"Processing every {process_every_n_frames}th frame")
        estimated_frames_to_process = total_frames // process_every_n_frames
    else:
        estimated_frames_to_process = total_frames
    
    # Process frames with proper error handling
    frame_count = 0
    print("Processing frames...")
    
    for frame in tqdm(frame_generator, total=total_frames, desc="Processing video"):
        frame_count += 1
        
        # Skip frames based on setting
        if process_every_n_frames > 1 and frame_count % process_every_n_frames != 0:
            frames_skipped += 1
            # Still write the frame to maintain video timing
            out.write(frame)
            continue
        
        try:
            frames_processed += 1
            
            # Scale down the frame if needed
            if scale_factor != 1.0:
                process_frame = cv2.resize(frame, (width, height))
            else:
                process_frame = frame
            
            # Predict the boxes with the model
            result = model.predict(process_frame, conf=0.3, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            
            # Scale back the detections if needed
            if scale_factor != 1.0:
                # Scale xyxy coordinates back to original size
                detections.xyxy = detections.xyxy / scale_factor
                
            # separate the detections
            goalkeepers_detections = detections[detections.class_id == MODEL_CLASSES["goalkepper"]]
            ball_detections = detections[detections.class_id == MODEL_CLASSES["ball"]]
            players_detections = detections[detections.class_id == MODEL_CLASSES["player"]]
            referee_detections = detections[detections.class_id == MODEL_CLASSES["referee"]]
            
            # Update stats
            detections_count["ball"] += len(ball_detections)
            detections_count["goalkeeper"] += len(goalkeepers_detections)
            detections_count["referee"] += len(referee_detections)

            # Applying Non-Maximum Suppression (NMS) to the players detections
            players_detections = players_detections.with_nms(threshold=0.5)
            
            # Assign Player Teams
            if is_first_frame:
                print("Assigning team colors...")
                team_assigner = Assigner()
                kmeans = team_assigner.assign_team_color(frame, players_detections)
                team_colors = team_assigner.team_colors
                is_first_frame = False
                print(f"Team colors assigned: {team_colors}")
            
            for object_ind, _ in enumerate(players_detections.class_id):
                player_color = team_assigner.get_player_color(frame, players_detections.xyxy[object_ind])
                team_id = team_assigner.get_player_team(frame, players_detections.xyxy[object_ind], kmeans)
                if team_id == 0:
                    players_detections.class_id[object_ind] = MODEL_CLASSES["team1"]
                    a = np.sqrt((player_color[0] - team_colors[1][0])**2 + (player_color[1] - team_colors[1][1])**2 + (player_color[2] - team_colors[1][2])**2)
                    if a > 110 and a < 180:
                        players_detections.class_id[object_ind] = MODEL_CLASSES["goalkepper"]
                elif team_id == 1:
                    players_detections.class_id[object_ind] = MODEL_CLASSES["team2"]
                    a = np.sqrt((player_color[0] - team_colors[2][0])**2 + (player_color[1] - team_colors[2][1])**2 + (player_color[2] - team_colors[2][2])**2)
                    if a > 110 and a < 180:
                        players_detections.class_id[object_ind] = MODEL_CLASSES["goalkepper"]
            
            goalkeepers_detections1 = players_detections[players_detections.class_id == MODEL_CLASSES["goalkepper"]]
            goalkeepers_detections = sv.Detections.merge([goalkeepers_detections, goalkeepers_detections1])
            
            team1_detections = players_detections[players_detections.class_id == MODEL_CLASSES["team1"]]
            team2_detections = players_detections[players_detections.class_id == MODEL_CLASSES["team2"]]
            
            # Update team detection counts
            detections_count["team1"] += len(team1_detections)
            detections_count["team2"] += len(team2_detections)
            
            players_detections = players_detections[players_detections.class_id != MODEL_CLASSES["goalkepper"]]
            # assign the ball to the closest player
            player_ind = assign_ball_to_player(players_detections, ball_detections.xyxy)
            all_players = players_detections
            if player_ind != -1:
                class_id_active_player = players_detections.class_id[player_ind]
                ball_posession[class_id_active_player] += 1
                all_players.class_id[player_ind] = MODEL_CLASSES["active_player"]
            elif class_id_active_player:
                ball_posession[class_id_active_player] += 1
            active_player_detection = all_players[all_players.class_id == MODEL_CLASSES["active_player"]]

            # adding a padding to the ball detection and active player
            if len(ball_detections.xyxy) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            if len(active_player_detection.xyxy) > 0:
                active_player_detection.xyxy = sv.pad_boxes(xyxy=active_player_detection.xyxy, px=10)

            # add tracker teams
            team1_detections = tracker.update_with_detections(detections=team1_detections)
            team2_detections = tracker1.update_with_detections(detections=team2_detections)

            # creating labels
            labels = {
                "labels_team1": [f"{tracker_id}" for tracker_id in team1_detections.tracker_id],
                "labels_team2": [f"{tracker_id}" for tracker_id in team2_detections.tracker_id],
                "labels_referee": ["ref"] * len(referee_detections),
                "labels_gk": ["GK"] * len(goalkeepers_detections)
            }
            
            # Annotate the frame
            all_detection = {
                "goalkeepers": goalkeepers_detections,
                "ball": ball_detections,
                "palyers": players_detections,
                "referee": referee_detections,
                "team1": team1_detections,
                "team2": team2_detections,
                "active_player": active_player_detection
            }
            
            # Apply enhanced annotations with optional heatmap
            annotated_frame = annotate_frames(
                frame, 
                all_detection, 
                labels, 
                ball_posession,
                show_heatmap=show_heatmap
            )
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)
            
            # Print progress at sensible intervals
            print_interval = max(1, estimated_frames_to_process // 10)  # Print about 10 updates
            if frames_processed % print_interval == 0:
                percentage = int(frames_processed / estimated_frames_to_process * 100)
                print(f"\nProgress: {percentage}% - Frame {frames_processed}/{estimated_frames_to_process}")
                print(f"Ball possession: Team1={ball_posession[MODEL_CLASSES['team1']]}, Team2={ball_posession[MODEL_CLASSES['team2']]}")
            
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            # Still write the original frame to keep video continuity
            out.write(frame)
            continue

    # Release resources
    out.release()
    
    # Print final statistics
    print(f"\n--- Processing Complete ---")
    print(f"Video saved as: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Frames processed: {frames_processed}")
    print(f"Frames skipped: {frames_skipped}")
    if frames_processed > 0:
        print(f"Average detections per processed frame:")
        print(f"  - Team 1 players: {detections_count['team1']/frames_processed:.2f}")
        print(f"  - Team 2 players: {detections_count['team2']/frames_processed:.2f}")
        print(f"  - Ball: {detections_count['ball']/frames_processed:.2f}")
        print(f"  - Referee: {detections_count['referee']/frames_processed:.2f}")
        print(f"  - Goalkeeper: {detections_count['goalkeeper']/frames_processed:.2f}")
    print(f"Final ball possession: Team1={ball_posession[MODEL_CLASSES['team1']]}, Team2={ball_posession[MODEL_CLASSES['team2']]}")
    
    # Calculate possession percentages
    total_possession = sum(ball_posession.values())
    if total_possession > 0:
        team1_pct = round(ball_posession[MODEL_CLASSES["team1"]] / total_possession * 100, 1)
        team2_pct = round(ball_posession[MODEL_CLASSES["team2"]] / total_possession * 100, 1)
        print(f"Possession percentages: Team1={team1_pct}%, Team2={team2_pct}%")
    
    # On Windows, suggest converting the video to a more compatible format using ffmpeg
    print("\nNote: If the output video has compatibility issues, consider using ffmpeg to convert:")
    print(f"ffmpeg -i {output_path} -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p {os.path.splitext(output_path)[0]}_compatible.mp4")

if __name__ == '__main__':
    main()