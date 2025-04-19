import cv2
import numpy as np
import os
import argparse
from sports.common.view import ViewTransformer
from functools import partial
import json

# Global variables for interactive point selection
points = []
current_point = 0
image_copy = None
window_name = "Select Source Points"
config_file = "source_points_config.json"

def click_event(event, x, y, flags, params):
    global points, current_point, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_point < 4:
            # Update the current point position
            points[current_point] = [x, y]
            
            # Update image with all points
            image = params["image"].copy()
            for i, point in enumerate(points):
                if point[0] != -1:  # If point is set
                    color = (0, 255, 0) if i == current_point else (0, 0, 255)
                    cv2.circle(image, (point[0], point[1]), 5, color, -1)
                    cv2.putText(image, f"P{i+1}", (point[0]+10, point[1]+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            image_copy = image.copy()
            cv2.imshow(window_name, image)
            
            # Move to next point
            current_point = (current_point + 1) % 4
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if image_copy is not None:
            img = image_copy.copy()
            cv2.putText(img, f"Position: ({x}, {y})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, img)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Delete the last point and go back
        if current_point > 0:
            current_point -= 1
            points[current_point] = [-1, -1]
            
            # Update image
            image = params["image"].copy()
            for i, point in enumerate(points):
                if point[0] != -1:  # If point is set
                    color = (0, 255, 0) if i == current_point else (0, 0, 255)
                    cv2.circle(image, (point[0], point[1]), 5, color, -1)
                    cv2.putText(image, f"P{i+1}", (point[0]+10, point[1]+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            image_copy = image.copy()
            cv2.imshow(window_name, image)

def get_source_points_interactive(image):
    global points, current_point, image_copy
    
    # Initialize points with placeholders
    points = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    current_point = 0
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, {"image": image})
    
    # Display instructions
    height, width, _ = image.shape
    instruction_img = np.zeros((100, width, 3), dtype=np.uint8)
    instructions = [
        "Click to set points in order: Bottom Left, Bottom Right, Top Left, Top Right",
        "Right-click to undo the last point",
        "Press Enter when all 4 points are set",
        "Press ESC to cancel and use default points"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(instruction_img, text, (10, 25 * (i+1)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Combine instruction image with the actual image
    display_img = np.vstack([instruction_img, image])
    image_copy = image.copy()
    
    cv2.imshow(window_name, display_img)
    
    # Wait for user to select points or press ESC to cancel
    all_points_set = False
    while not all_points_set:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            # Check if all points are set
            if all(p[0] != -1 for p in points):
                all_points_set = True
            else:
                print("Please set all 4 points before proceeding")
        
        elif key == 27:  # ESC key
            print("Using default points")
            height, width, _ = image.shape
            points = [
                [int(width * 0.2), int(height * 0.7)],   # Bottom left corner of field
                [int(width * 0.8), int(height * 0.7)],   # Bottom right corner of field
                [int(width * 0.1), int(height * 0.25)],  # Top left corner of field
                [int(width * 0.9), int(height * 0.25)]   # Top right corner of field
            ]
            all_points_set = True
    
    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.float32)

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
    parser = argparse.ArgumentParser(description='ViewTransformer Example')
    parser.add_argument('--input', type=str, default='input_video/08fd33_4.mp4', 
                      help='Path to input video or image (relative to local_exec folder)')
    parser.add_argument('--output_dir', type=str, default='output_video', 
                      help='Output directory for transformed images/video')
    parser.add_argument('--frame', type=int, default=0, 
                      help='Frame number to extract from video (if using video)')
    parser.add_argument('--mode', type=str, choices=['single_frame', 'video', 'interactive'], default='interactive',
                      help='Process a single frame, full video, or interactive mode')
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
        
        if args.mode == 'interactive':
            interactive_transform(img, output_dir, args.load_points)
        else:
            transform_single_image(img, output_dir)
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
        
        if args.mode == 'interactive':
            # Extract a frame for interactive point selection
            frame_num = min(args.frame, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                print(f"Extracted frame {frame_num} for interactive point selection")
                interactive_transform(frame, output_dir, args.load_points, cap, fps, total_frames)
            else:
                print(f"Error: Could not read frame {frame_num}")
        elif args.mode == 'single_frame':
            # Process a single frame
            frame_num = min(args.frame, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                print(f"Extracted frame {frame_num}")
                transform_single_image(frame, output_dir, frame_num)
            else:
                print(f"Error: Could not read frame {frame_num}")
        else:
            # Process the entire video
            transform_video(cap, output_dir, fps, total_frames)
            
        cap.release()

def interactive_transform(image, output_dir, load_saved_points=False, cap=None, fps=None, total_frames=None):
    """Interactive transformation with manual source point selection"""
    height, width, _ = image.shape
    
    # Define output dimensions for the projection
    proj_width, proj_height = 800, 500
    
    # Load saved points or get them interactively
    if load_saved_points:
        saved_points = load_source_points(output_dir)
        if saved_points is not None:
            source_points = saved_points
            print("Loaded previously saved source points")
        else:
            print("No saved points found, please select points manually")
            source_points = get_source_points_interactive(image)
    else:
        source_points = get_source_points_interactive(image)
    
    # Save the selected points for future use
    save_source_points(source_points, output_dir)
    
    # Draw points on a copy of the image for visualization
    points_img = image.copy()
    for i, point in enumerate(source_points):
        cv2.circle(points_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
        cv2.putText(points_img, f"P{i+1}", (int(point[0])+5, int(point[1])+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save image with points
    points_path = os.path.join(output_dir, "source_points_selected.jpg")
    cv2.imwrite(points_path, points_img)
    print(f"Image with selected source points saved to: {points_path}")
    
    # Define target points (rectangular view)
    target_points = np.array([
        [0, proj_height],             # Bottom left
        [proj_width, proj_height],    # Bottom right
        [0, 0],                       # Top left
        [proj_width, 0]               # Top right
    ], dtype=np.float32)
    
    # Create the ViewTransformer
    try:
        print("Creating transformation...")
        transformer = ViewTransformer(source_points, target_points)
        
        # Transform the image
        birds_eye = transformer.transform_image(image, (proj_width, proj_height))
        
        # Save the transformed image
        birds_eye_path = os.path.join(output_dir, "interactive_birds_eye_view.jpg")
        cv2.imwrite(birds_eye_path, birds_eye)
        print(f"Birds-eye view saved to: {birds_eye_path}")
        
        # Display the result
        cv2.namedWindow("Birds-Eye View")
        cv2.imshow("Birds-Eye View", birds_eye)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # If video capture is provided, ask if user wants to process the entire video
        if cap is not None:
            choice = input("Do you want to process the entire video with these points? (y/n): ")
            if choice.lower() == 'y':
                # Reset video to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Create output video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = os.path.join(output_dir, "interactive_birds_eye_video.mp4")
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
        print(f"Error creating ViewTransformer: {e}")

def transform_single_image(image, output_dir, frame_num=None):
    """Transform a single image using ViewTransformer and save different perspectives"""
    height, width, _ = image.shape
    
    # Output file naming
    suffix = f"_frame{frame_num}" if frame_num is not None else ""
    
    # Display and save original image
    output_path = os.path.join(output_dir, f"original{suffix}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Original image saved to: {output_path}")
    
    # 1. Create a birds-eye view (top-down perspective)
    # Define output dimensions for the projection
    proj_width, proj_height = 800, 500
    
    # Define source points (corners of the field in the original image)
    # These are approximate and should be adjusted for your specific video
    source_points = np.array([
        [int(width * 0.2), int(height * 0.7)],   # Bottom left corner of field
        [int(width * 0.8), int(height * 0.7)],   # Bottom right corner of field
        [int(width * 0.1), int(height * 0.25)],  # Top left corner of field
        [int(width * 0.9), int(height * 0.25)]   # Top right corner of field
    ], dtype=np.float32)
    
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

def transform_video(cap, output_dir, fps, total_frames):
    """Transform an entire video using ViewTransformer"""
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    height, width, _ = first_frame.shape
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Define projection dimensions
    proj_width, proj_height = 800, 500
    
    # Define source points (corners of the field in the original image)
    # These are approximate and should be adjusted for your specific video
    source_points = np.array([
        [int(width * 0.2), int(height * 0.7)],   # Bottom left corner of field
        [int(width * 0.8), int(height * 0.7)],   # Bottom right corner of field
        [int(width * 0.1), int(height * 0.25)],  # Top left corner of field
        [int(width * 0.9), int(height * 0.25)]   # Top right corner of field
    ], dtype=np.float32)
    
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

if __name__ == "__main__":
    main()