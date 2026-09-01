import cv2
import numpy as np
import torch
import time
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import threading

class GoalDetector:
    def __init__(self, model_path=None, conf_threshold=0.25, video_path=None, debug_mode=False, visualization=True):
        """
        Initialize the Goal Detector class
        
        Args:
            model_path: Path to the YOLOv5 model weights
            conf_threshold: Confidence threshold for detections
            video_path: Path to the video file for analysis
            debug_mode: Enable debug mode for visualization and diagnostics
            visualization: Enable live visualization
        """
        self.conf_threshold = conf_threshold
        self.video_path = video_path
        self.detected_goals = []
        self.debug_mode = debug_mode
        self.visualization = visualization
        
        # Create debug directory if in debug mode
        if self.debug_mode:
            self.debug_dir = "debug_frames"
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
        
        # Load YOLOv5 model
        if model_path:
            print(f"Loading custom model from {model_path}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            print("Loading YOLOv5s model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # Set model parameters
        self.model.conf = conf_threshold  # Confidence threshold
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Initialize tracking variables
        self.ball_positions = []  # To track ball positions over time
        self.ball_position_window = 10  # Number of frames to track for velocity
        
        # Visualization variables
        self.visualization_frame = None
        self.visualization_lock = threading.Lock()
        self.visualization_active = True
    
    def detect_objects(self, frame):
        """
        Detect objects (ball, players, goalpost) in a frame
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            Dictionary with detected objects
        """
        # Perform object detection
        results = self.model(frame)
        
        # Convert results to pandas dataframe
        detections = results.pandas().xyxy[0]
        
        # Create dictionary to hold detection results
        detected_objects = {
            'ball': None,
            'goalpost': None,
            'players': []
        }
        
        # Filter detections
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = float(detection['confidence'])
            class_name = detection['name']
            
            # Store detection based on class name
            if class_name == 'sports ball' and confidence > self.conf_threshold:
                detected_objects['ball'] = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                }
                
                # Track ball position for velocity calculation
                if detected_objects['ball']:
                    self.ball_positions.append(detected_objects['ball']['center'])
                    # Keep only the last N positions
                    if len(self.ball_positions) > self.ball_position_window:
                        self.ball_positions.pop(0)
                
            elif class_name == 'person' and confidence > self.conf_threshold:
                detected_objects['players'].append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return detected_objects
    
    def detect_goalpost(self, frame):
        """
        Detect goalposts using color thresholding and contour analysis
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            List of goalpost bounding boxes
        """
        # Convert to HSV for better color thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define multiple ranges for white/light color (goalpost)
        # Broader range to catch more potential goalposts
        lower_white1 = np.array([0, 0, 180])
        upper_white1 = np.array([180, 40, 255])
        
        # Create mask and apply morphological operations
        mask1 = cv2.inRange(hsv, lower_white1, upper_white1)
        
        # Combine masks if you have multiple ranges
        mask = mask1
        
        # Process mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Save mask for debugging if in debug mode
        if self.debug_mode:
            cv2.imwrite(os.path.join(self.debug_dir, f"mask_{time.time()}.jpg"), mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        goalpost_candidates = []
        
        # Filter contours to find goalpost-like shapes with more relaxed criteria
        for contour in contours:
            # Check if contour area is large enough
            if cv2.contourArea(contour) > 500:  # Lowered threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if shape could be part of a goalpost (relaxed criteria)
                if h > w * 1.5 or w > h * 1.5:  # Either tall and thin OR wide and short
                    goalpost_candidates.append((x, y, x + w, y + h))
        
        return goalpost_candidates
    
    def calculate_ball_velocity(self):
        """
        Calculate the velocity of the ball based on recent position history
        
        Returns:
            Velocity magnitude (pixels per frame)
        """
        if len(self.ball_positions) < 2:
            return 0
        
        # Get the most recent positions
        pos1 = self.ball_positions[-2]
        pos2 = self.ball_positions[-1]
        
        # Calculate displacement
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        # Calculate velocity magnitude
        velocity = np.sqrt(dx*dx + dy*dy)
        
        return velocity
    
    def is_ball_in_goalpost_area(self, ball, goalposts, frame_width, frame_height):
        """
        Check if the ball is in the goalpost area
        
        Args:
            ball: Ball detection data
            goalposts: List of goalpost bounding boxes
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Boolean indicating if ball is in goalpost area
        """
        if ball is None or not goalposts:
            return False
        
        ball_center_x, ball_center_y = ball['center']
        
        for goalpost in goalposts:
            x1, y1, x2, y2 = goalpost
            
            # Expand goalpost area a bit
            expanded_x1 = max(0, x1 - 50)  # Increased expansion
            expanded_x2 = min(frame_width, x2 + 50)
            expanded_y1 = max(0, y1 - 50)
            expanded_y2 = min(frame_height, y2 + 50)
            
            # Check if ball center is inside expanded goalpost area
            if expanded_x1 <= ball_center_x <= expanded_x2 and expanded_y1 <= ball_center_y <= expanded_y2:
                return True
        
        return False
    
    def detect_scene_change(self, current_frame, previous_frame):
        """
        Detect if there was a scene change (useful for goal celebrations)
        
        Args:
            current_frame: Current video frame
            previous_frame: Previous video frame
            
        Returns:
            Boolean indicating if a scene change was detected
        """
        if previous_frame is None:
            return False
        
        # Convert frames to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(current_gray, previous_gray)
        
        # Calculate mean difference
        mean_diff = np.mean(diff)
        
        # Check if mean difference is above threshold
        return mean_diff > 20  # Adjust threshold as needed
    
    def count_players_in_center(self, players, frame_width, frame_height):
        """
        Count how many players are in the center of frame (for celebration detection)
        
        Args:
            players: List of player detections
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Number of players in center area
        """
        center_players = 0
        
        # Define center area (middle 50% of the frame)
        center_x1 = frame_width * 0.25
        center_x2 = frame_width * 0.75
        center_y1 = frame_height * 0.25
        center_y2 = frame_height * 0.75
        
        for player in players:
            player_center_x, player_center_y = player['center']
            
            if (center_x1 <= player_center_x <= center_x2 and 
                center_y1 <= player_center_y <= center_y2):
                center_players += 1
        
        return center_players
    
    def detect_goal_event(self, current_frame, previous_frame, previous_detections, current_detections, frame_width, frame_height):
        """
        Detect if a goal event has occurred using multiple methods
        
        Args:
            current_frame: Current video frame
            previous_frame: Previous video frame
            previous_detections: Object detections from previous frame
            current_detections: Object detections from current frame
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Boolean indicating if a goal was detected
        """
        # Get current and previous ball
        current_ball = current_detections.get('ball', None)
        previous_ball = previous_detections.get('ball', None)
        
        # Detect goalposts using color thresholding
        goalposts = self.detect_goalpost(current_frame)
        
        # Calculate ball velocity
        ball_velocity = self.calculate_ball_velocity()
        
        # Detect scene change
        scene_change = self.detect_scene_change(current_frame, previous_frame)
        
        # Count players in center of frame
        center_players = self.count_players_in_center(current_detections.get('players', []), frame_width, frame_height)
        
        # Method 1: Ball near goalpost
        ball_near_goalpost = current_ball and self.is_ball_in_goalpost_area(current_ball, goalposts, frame_width, frame_height)
        
        # Method 2: Scene change (camera cut) after ball was in motion
        goal_by_scene_change = scene_change and ball_velocity > 10 and len(self.ball_positions) >= 3
        
        # Method 3: Many players gathered in center (celebration)
        celebration_detected = center_players >= 3
        
        # Combined goal detection logic
        if ball_near_goalpost and (ball_velocity > 5 or celebration_detected):
            return True
        
        if goal_by_scene_change and celebration_detected:
            return True
        
        return False
    
    def create_visualization_frame(self, frame, detections, goalposts, frame_number, velocity=0, goal_detected=False):
        """
        Create a visualization frame with annotations
        
        Args:
            frame: The video frame
            detections: Dictionary of detected objects
            goalposts: List of goalpost bounding boxes
            frame_number: Current frame number
            velocity: Ball velocity
            goal_detected: Whether a goal was detected in this frame
            
        Returns:
            Annotated frame for visualization
        """
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        # Add YOLO Detection Demo text at the top
        cv2.putText(vis_frame, "YOLO Soccer Goal Detection Demo", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ball if detected
        if detections.get('ball'):
            x1, y1, x2, y2 = detections['ball']['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Ball {detections['ball']['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ball path
            if len(self.ball_positions) > 1:
                for i in range(1, len(self.ball_positions)):
                    cv2.line(vis_frame, 
                            (self.ball_positions[i-1][0], self.ball_positions[i-1][1]),
                            (self.ball_positions[i][0], self.ball_positions[i][1]),
                            (0, 255, 255), 2)
        
        # Draw players
        for i, player in enumerate(detections.get('players', [])):
            x1, y1, x2, y2 = player['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw goalposts
        for gp in goalposts:
            x1, y1, x2, y2 = gp
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_frame, "Goalpost", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add info panel at the bottom
        panel_height = 120
        panel = np.zeros((panel_height, vis_frame.shape[1], 3), dtype=np.uint8)
        
        # Add info text to panel
        cv2.putText(panel, f"Frame: {frame_number}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, f"Ball Velocity: {velocity:.2f} px/frame", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, f"Players Detected: {len(detections.get('players', []))}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(panel, f"Goalposts Detected: {len(goalposts)}", 
                   (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add detected goal indicator if goal was detected
        if goal_detected:
            cv2.putText(panel, "GOAL DETECTED!", 
                       (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Combine frame and panel
        combined = np.vstack((vis_frame, panel))
        
        return combined
    
    def visualization_thread(self):
        """
        Thread function to handle visualization window
        """
        window_name = "Soccer Goal Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.visualization_active:
            with self.visualization_lock:
                if self.visualization_frame is not None:
                    cv2.imshow(window_name, self.visualization_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                self.visualization_active = False
        
        cv2.destroyAllWindows()
    
    def process_video(self):
        """
        Process the video to detect goals
        
        Returns:
            List of timestamps where goals were detected
        """
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video FPS: {fps}")
        print(f"Video Dimensions: {frame_width}x{frame_height}")
        print(f"Total Frames: {total_frames}")
        
        # Initialize variables
        previous_detections = {'ball': None, 'goalpost': None, 'players': []}
        previous_frame = None
        frame_count = 0
        goal_cooldown = int(fps * 5)  # 5 seconds cooldown after a goal
        cooldown_counter = 0
        
        # Create output directory for goal clips if it doesn't exist
        output_dir = "goal_clips"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Statistics tracking
        stats = {
            'frames_with_ball': 0,
            'frames_with_players': 0,
            'frames_with_goalposts': 0,
            'total_processed_frames': 0
        }
        
        # Start visualization thread if visualization is enabled
        if self.visualization:
            vis_thread = threading.Thread(target=self.visualization_thread)
            vis_thread.daemon = True
            vis_thread.start()
        
        # Process frames
        while cap.isOpened() and (self.visualization_active or not self.visualization):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Progress update every 100 frames
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Update statistics
            stats['total_processed_frames'] += 1
            
            # Skip frames if in cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
                frame_count += 1
                previous_frame = frame.copy()
                continue
            
            # Detect objects in current frame
            current_detections = self.detect_objects(frame)
            
            # Update statistics
            if current_detections.get('ball'):
                stats['frames_with_ball'] += 1
            if current_detections.get('players'):
                stats['frames_with_players'] += 1
            
            # Detect goalposts
            goalposts = self.detect_goalpost(frame)
            if goalposts:
                stats['frames_with_goalposts'] += 1
            
            # Calculate ball velocity
            ball_velocity = self.calculate_ball_velocity()
            
            # Check for goal event
            goal_detected = False
            if self.detect_goal_event(frame, previous_frame, previous_detections, current_detections, frame_width, frame_height):
                goal_detected = True
                
                # Get timestamp of the goal
                timestamp = frame_count / fps
                formatted_time = time.strftime('%M:%S', time.gmtime(timestamp))
                
                print(f"Goal detected at {formatted_time} (frame {frame_count})")
                
                # Save goal event details
                self.detected_goals.append({
                    'timestamp': timestamp,
                    'formatted_time': formatted_time,
                    'frame_number': frame_count
                })
                
                # Save a clip of the goal
                goal_clip_path = os.path.join(output_dir, f"goal_{len(self.detected_goals)}.mp4")
                self.save_goal_clip(cap, frame_count, fps, goal_clip_path)
                
                # Set cooldown after a goal is detected
                cooldown_counter = goal_cooldown
            
            # Create visualization frame if visualization is enabled
            if self.visualization:
                vis_frame = self.create_visualization_frame(
                    frame, current_detections, goalposts, frame_count, ball_velocity, goal_detected
                )
                
                # Update visualization frame
                with self.visualization_lock:
                    self.visualization_frame = vis_frame
            
            # Save debug frame if in debug mode
            if self.debug_mode and (frame_count % 30 == 0 or goal_detected):
                self.debug_save_frame(frame, current_detections, goalposts, frame_count, ball_velocity)
            
            # Update previous detections for next frame
            previous_detections = current_detections
            previous_frame = frame.copy()
            frame_count += 1
        
        # Clean up
        cap.release()
        
        # Stop visualization thread
        if self.visualization:
            self.visualization_active = False
            time.sleep(0.5)  # Give thread time to exit
        
        # Print statistics after processing
        print("\nDetection Statistics:")
        print(f"Total processed frames: {stats['total_processed_frames']}")
        if stats['total_processed_frames'] > 0:
            print(f"Frames with ball: {stats['frames_with_ball']} ({stats['frames_with_ball']/stats['total_processed_frames']*100:.2f}%)")
            print(f"Frames with players: {stats['frames_with_players']} ({stats['frames_with_players']/stats['total_processed_frames']*100:.2f}%)")
            print(f"Frames with goalposts: {stats['frames_with_goalposts']} ({stats['frames_with_goalposts']/stats['total_processed_frames']*100:.2f}%)")
        
        # Plot statistics if in debug mode
        if self.debug_mode:
            self.plot_statistics(stats)
        
        return self.detected_goals
    
    def debug_save_frame(self, frame, detections, goalposts, frame_number, velocity=0):
        """
        Save debug frames with annotations to visualize detections
        
        Args:
            frame: The video frame
            detections: Dictionary of detected objects
            goalposts: List of goalpost bounding boxes
            frame_number: Current frame number
            velocity: Ball velocity
        """
        if not self.debug_mode:
            return
        
        # Create visualization frame
        debug_frame = self.create_visualization_frame(
            frame, detections, goalposts, frame_number, velocity
        )
        
        # Save the debug frame
        output_path = os.path.join(self.debug_dir, f"frame_{frame_number:06d}.jpg")
        cv2.imwrite(output_path, debug_frame)
    
    def save_goal_clip(self, cap, goal_frame, fps, output_path, clip_duration=10):
        """
        Save a clip of the detected goal
        
        Args:
            cap: Video capture object
            goal_frame: Frame number where goal was detected
            fps: Frames per second of the video
            output_path: Path to save the clip
            clip_duration: Duration of the clip in seconds
        """
        # Calculate start and end frames
        frames_per_clip = int(fps * clip_duration)
        start_frame = max(0, goal_frame - int(frames_per_clip * 0.7))  # 70% before the goal
        end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), start_frame + frames_per_clip)
        
        # Set video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Save the clip
        current_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)
        
        # Release resources
        out.release()
        print(f"Goal clip saved to {output_path}")
    
    def save_visualization_video(self, processed_frames, output_path, fps):
        """
        Save a video with all the visualization frames
        
        Args:
            processed_frames: List of processed frames
            output_path: Path to save the video
            fps: Frames per second
        """
        if not processed_frames:
            return
        
        # Get frame dimensions
        height, width = processed_frames[0].shape[:2]
        
        # Set video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in processed_frames:
            out.write(frame)
        
        # Release resources
        out.release()
        print(f"Visualization video saved to {output_path}")
    
    def plot_statistics(self, stats):
        """
        Plot detection statistics
        
        Args:
            stats: Dictionary containing detection statistics
        """
        if stats['total_processed_frames'] == 0:
            return
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Data for plotting
        categories = ['Ball', 'Players', 'Goalposts']
        percentages = [
            stats['frames_with_ball'] / stats['total_processed_frames'] * 100,
            stats['frames_with_players'] / stats['total_processed_frames'] * 100,
            stats['frames_with_goalposts'] / stats['total_processed_frames'] * 100
        ]
        
        # Create bar plot
        bars = ax.bar(categories, percentages)
        
        # Add labels and title
        ax.set_xlabel('Detection Type')
        ax.set_ylabel('Percentage of Frames (%)')
        ax.set_title('Object Detection Statistics')
        ax.set_ylim(0, 100)
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentages[i]:.1f}%',
                   ha='center', va='bottom', rotation=0)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.debug_dir, 'detection_statistics.png'))
        plt.close()
    
    def display_results(self):
        """
        Display results of goal detection
        """
        if not self.detected_goals:
            print("No goals detected in the video.")
            return
        
        print("\n" + "="*50)
        print("GOAL DETECTION RESULTS")
        print("="*50)
        
        for i, goal in enumerate(self.detected_goals, 1):
            print(f"Goal {i}: Timestamp {goal['formatted_time']} (Frame {goal['frame_number']})")
        
        print("="*50)


def main():
    """
    Main function to run the goal detector
    """
    parser = argparse.ArgumentParser(description='Soccer Goal Detection using Computer Vision')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLOv5 model weights (optional)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for visualization')
    parser.add_argument('--no-viz', action='store_true', help='Disable live visualization')
    
    args = parser.parse_args()
    
    # Create goal detector instance
    goal_detector = GoalDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        video_path=args.video,
        debug_mode=args.debug,
        visualization=not args.no_viz
    )
    
    # Process video
    start_time = time.time()
    goal_detector.process_video()
    end_time = time.time()
    
    # Display results
    goal_detector.display_results()
    
    # Print processing time
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")


if __name__ == "__main__":
    main()
