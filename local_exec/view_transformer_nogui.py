import cv2
import numpy as np
import os
import argparse
from sports.common.view import ViewTransformer
import json

# Configuration file for source points
config_file = "source_points_config.json"

def save_source_points(points, output_dir):
    """Save the selected source points to a configuration file"""
    config_path = os.path.join(output_dir, config_file)
    with open(config_path, 'w') as f:
        json.dump({'source_points': points.tolist()}, f)
    print(f"Source points saved to {config_path}")

def load_source_points(output_dir):
    """Load source points from a configuration file if it exists"""
    config_path = os.path.join(output_dir, config_file)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return np.array(config['source_points'], dtype=np.float32)
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ViewTransformer Example (No GUI)')
    parser.add_argument('--input', type=str, default='input_video/08fd33_4.mp4', 
                      help='Path to input video or image (relative to local_exec folder)')
    parser.add_argument('--output_dir', type=str, default='output_video', 
                      help='Output directory for transformed images/video')
    parser.add_argument('--frame', type=int, default=0, 
                      help='Frame number to extract from video (if using video)')
    parser.add_argument('--mode', type=str, choices=['single_frame', 'video'], default='single_frame',
                      help='Process a single frame or full video')
    parser.add_argument('--points', type=str, default=None,
                      help='Comma-separated list of points in format "x1,y1,x2,y2,x3,y3,x4,y4"')
    parser.add_argument('--load_points', action='store_true', 
                      help='Load previously saved source points')
    args = parser.parse_args()
    
    # Make paths absolute
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, args.input)
    output_dir = os.path.join(current_dir, args.output_dir)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if input is an image or video
    is_image = input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
    
    if is_image:
        # Read the image
        print(f"Reading image: {input_path}")
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return
        transform_single_image(img, output_dir, args.points, args.load_points)
    else:
        # Video processing
        print(f"Opening video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video dimensions: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        if args.mode == 'single_frame':
            # Process a single frame
            frame_num = min(args.frame, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                print(f"Extracted frame {frame_num}")
                transform_single_image(frame, output_dir, args.points, args.load_points, frame_num)
            else:
                print(f"Error: Could not read frame {frame_num}")
        else:
            # Process the entire video
            transform_video(cap, output_dir, fps, total_frames, args.points, args.load_points)
            
        cap.release()

def transform_single_image(image, output_dir, points_str=None, load_saved_points=False, frame_num=None):
    """Transform a single image using ViewTransformer and save different perspectives"""
    height, width, _ = image.shape
    
    # Output file naming
    suffix = f"_frame{frame_num}" if frame_num is not None else ""
    
    # Display and save original image
    output_path = os.path.join(output_dir, f"original{suffix}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Original image saved to: {output_path}")
    
    # Define output dimensions for the projection
    proj_width, proj_height = 800, 500
    
    # Get source points
    source_points = get_source_points(image, points_str, load_saved_points, output_dir)
    
    # Draw points on a copy of the image for visualization
    points_img = image.copy()
    for i, point in enumerate(source_points):
        cv2.circle(points_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        cv2.putText(points_img, f"P{i+1}", (int(point[0])+5, int(point[1])+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save image with points
    points_path = os.path.join(output_dir, f"source_points{suffix}.jpg")
    cv2.imwrite(points_path, points_img)
    print(f"Image with source points saved to: {points_path}")
    
    # Define target points (rectangular view)
    target_points = np.array([
        [0, proj_height],             # Bottom left
        [proj_width, proj_height],    # Bottom right
        [0, 0],                       # Top left
        [proj_width, 0]               # Top right
    ], dtype=np.float32)
    
    # Create the ViewTransformer
    try:
        print("Creating birds-eye view transformation...")
        transformer = ViewTransformer(source_points, target_points)
        
        # Transform the image
        birds_eye = transformer.transform_image(image, (proj_width, proj_height))
        
        # Save the transformed image
        birds_eye_path = os.path.join(output_dir, f"birds_eye_view{suffix}.jpg")
        cv2.imwrite(birds_eye_path, birds_eye)
        print(f"Birds-eye view saved to: {birds_eye_path}")
        
    except ValueError as e:
        print(f"Error creating ViewTransformer: {e}")
    
    # 2. Create a corrected perspective (fix skew)
    try:
        print("Creating corrected perspective transformation...")
        # Here we try to keep the image size but correct the perspective
        corrected_points = np.array([
            [0, height],              # Bottom left
            [width, height],          # Bottom right
            [0, 0],                   # Top left
            [width, 0]                # Top right
        ], dtype=np.float32)
        
        corrector = ViewTransformer(source_points, corrected_points)
        corrected = corrector.transform_image(image, (width, height))
        
        # Save the transformed image
        corrected_path = os.path.join(output_dir, f"corrected_perspective{suffix}.jpg")
        cv2.imwrite(corrected_path, corrected)
        print(f"Corrected perspective saved to: {corrected_path}")
        
    except ValueError as e:
        print(f"Error creating corrector: {e}")

def transform_video(cap, output_dir, fps, total_frames, points_str=None, load_saved_points=False):
    """Transform an entire video using ViewTransformer"""
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    height, width, _ = first_frame.shape
    
    # Get source points
    source_points = get_source_points(first_frame, points_str, load_saved_points, output_dir)
    
    # Draw points on first frame for visualization
    points_img = first_frame.copy()
    for i, point in enumerate(source_points):
        cv2.circle(points_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        cv2.putText(points_img, f"P{i+1}", (int(point[0])+5, int(point[1])+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save image with points
    points_path = os.path.join(output_dir, "source_points_video.jpg")
    cv2.imwrite(points_path, points_img)
    print(f"First frame with source points saved to: {points_path}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Define projection dimensions
    proj_width, proj_height = 800, 500
    
    # Define target points for birds-eye view
    target_points = np.array([
        [0, proj_height],             # Bottom left
        [proj_width, proj_height],    # Bottom right
        [0, 0],                       # Top left
        [proj_width, 0]               # Top right
    ], dtype=np.float32)
    
    try:
        # Create transformer for birds-eye view
        transformer = ViewTransformer(source_points, target_points)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, "birds_eye_video.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (proj_width, proj_height))
        
        if not out.isOpened():
            print("Error: Could not create output video writer")
            return
            
        print(f"Processing video frames and creating birds-eye view...")
        
        # Process frames with progress indicator
        from tqdm import tqdm
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Transform the frame
            birds_eye = transformer.transform_image(frame, (proj_width, proj_height))
            
            # Write to output
            out.write(birds_eye)
            
        # Release resources
        out.release()
        print(f"Transformed video saved to: {output_path}")
        
    except ValueError as e:
        print(f"Error in video transformation: {e}")

def get_source_points(image, points_str=None, load_saved_points=False, output_dir=None):
    """Get source points from string, saved config, or use defaults"""
    height, width, _ = image.shape
    
    # 1. Try to use points from command line argument
    if points_str:
        try:
            # Parse comma-separated list of points: "x1,y1,x2,y2,x3,y3,x4,y4"
            point_values = [int(x) for x in points_str.split(',')]
            if len(point_values) != 8:
                raise ValueError("Points string must contain exactly 8 values")
            
            points = np.array([
                [point_values[0], point_values[1]],  # Bottom left
                [point_values[2], point_values[3]],  # Bottom right
                [point_values[4], point_values[5]],  # Top left
                [point_values[6], point_values[7]]   # Top right
            ], dtype=np.float32)
            print("Using source points from command line argument")
            
            # Save these points for future use
            if output_dir:
                save_source_points(points, output_dir)
            
            return points
        except (ValueError, IndexError) as e:
            print(f"Error parsing points string: {e}")
            print("Falling back to other methods...")
    
    # 2. Try to load saved points
    if load_saved_points and output_dir:
        saved_points = load_source_points(output_dir)
        if saved_points is not None:
            print("Using previously saved source points")
            return saved_points
    
    # 3. Use default points based on image dimensions
    print("Using default calculated source points")
    return np.array([
        [int(width * 0.2), int(height * 0.7)],   # Bottom left corner of field
        [int(width * 0.8), int(height * 0.7)],   # Bottom right corner of field
        [int(width * 0.1), int(height * 0.25)],  # Top left corner of field
        [int(width * 0.9), int(height * 0.25)]   # Top right corner of field
    ], dtype=np.float32)

if __name__ == "__main__":
    main()