import cv2
import numpy as np
import argparse
from sports.common.view import ViewTransformer
from utils import get_number_of_frames, get_frames
import os
from config import VIDEO_SRC

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate 2D perspective projection video')
    parser.add_argument('--input', type=str, default=VIDEO_SRC, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_2d_projection.mp4', help='Path to output video')
    parser.add_argument('--width', type=int, default=800, help='Width of output projection field')
    parser.add_argument('--height', type=int, default=500, help='Height of output projection field')
    args = parser.parse_args()
    
    # Get video path and output path
    video_src = args.input
    output_path = os.path.join(os.path.dirname(video_src), '..', 'output_video', args.output)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get video information
    total_frames, fps = get_number_of_frames(video_src)
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Get the frames
    frame_generator = get_frames(video_src)
    
    # Get the first frame to determine the video dimensions
    first_frame = next(frame_generator)
    original_height, original_width, _ = first_frame.shape
    print(f"Video dimensions: {original_width}x{original_height}")
    
    # Define the output resolution for the 2D projection
    projection_width = args.width
    projection_height = args.height
    
    # Define source points (you need to adjust these to match your specific video)
    # These points should be corners of the field in the original camera view
    # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    source_points = np.array([
        [int(original_width * 0.2), int(original_height * 0.7)],  # Bottom left corner of field
        [int(original_width * 0.8), int(original_height * 0.7)],  # Bottom right corner of field
        [int(original_width * 0.1), int(original_height * 0.25)], # Top left corner of field
        [int(original_width * 0.9), int(original_height * 0.25)]  # Top right corner of field
    ], dtype=np.float32)
    
    # Define target points for a rectangular top-down view
    target_points = np.array([
        [0, projection_height],                  # Bottom left
        [projection_width, projection_height],   # Bottom right
        [0, 0],                                  # Top left
        [projection_width, 0]                    # Top right
    ], dtype=np.float32)
    
    # Create ViewTransformer with source and target points
    print("Creating homography transformation...")
    try:
        transformer = ViewTransformer(source_points, target_points)
    except ValueError as e:
        print(f"Error creating ViewTransformer: {e}")
        return
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (projection_width, projection_height))
    if not out.isOpened():
        raise Exception("Failed to create video writer. Check that the output directory exists and is writable.")
    
    print(f"Processing video and generating 2D projection to {output_path}...")
    
    # Process first frame
    projection = transformer.transform_image(first_frame, (projection_width, projection_height))
    out.write(projection)
    
    # Process remaining frames with progress bar
    from tqdm import tqdm
    for frame in tqdm(frame_generator, total=total_frames - 1):
        projection = transformer.transform_image(frame, (projection_width, projection_height))
        out.write(projection)
    
    # Release resources
    out.release()
    print(f"2D perspective projection video created at {output_path}")
    
    # Display instructions for fine-tuning the projection
    print("\nNote: You may need to fine-tune the source_points in the script to match your specific video.")
    print("The current points are estimates and may not perfectly align with the football field.")

if __name__ == "__main__":
    main()