import cv2
import os

class VideoProcessor:
    """Class to handle video processing operations"""
    
    def __init__(self, input_path, output_path):
        """Initialize the video processor
        
        Args:
            input_path (str): Path to the input video
            output_path (str): Path to save the output video
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.out = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
    
    def setup_video(self):
        """Set up video capture and writer
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.input_path}")
            return False
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        
        return True
    
    def read_frame(self):
        """Read a frame from the video
        
        Returns:
            tuple: (success, frame)
        """
        return self.cap.read()
    
    def write_frame(self, frame):
        """Write a frame to the output video
        
        Args:
            frame: The frame to write
        """
        self.out.write(frame)
    
    def show_frame(self, frame, window_name="Accident Detection"):
        """Display a frame
        
        Args:
            frame: The frame to display
            window_name (str): Name of the display window
        """
        cv2.imshow(window_name, frame)
    
    def check_key_press(self, key='q'):
        """Check if a key was pressed
        
        Args:
            key (str): Key to check for
            
        Returns:
            bool: True if key was pressed, False otherwise
        """
        return cv2.waitKey(1) & 0xFF == ord(key)
    
    def release(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
    
    def draw_detections(self, frame, truck_box, container_boxes):
        """Draw detection labels on the frame
        
        Args:
            frame: Video frame to draw on
            truck_box: Truck bounding box
            container_boxes: List of container bounding boxes
            
        Returns:
            frame: Frame with labels drawn
        """
        # Draw truck with label 'Truck' - but only the label, no bounding box
        if truck_box:
            x1, y1, x2, y2 = map(int, truck_box)
            color = (0, 255, 0)  # Always keep green color
            
            # Only draw the label, not the bounding box
            cv2.putText(frame, "Truck", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw container with letter 'C' - but only the label, no bounding box
        if container_boxes:
            for box in container_boxes:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)  # Always keep green color
                
                # Only draw the label, not the bounding box
                cv2.putText(frame, "C", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame
    
    def draw_accident_warning(self, frame):
        """Draw accident warning on the frame
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            frame: Frame with warning drawn
        """
        # Create a semi-transparent overlay for the accident warning
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 30), (600, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Add warning text
        cv2.putText(frame, "ACCIDENT DETECTED", (60, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
    
    def save_screenshot(self, frame, directory, filename):
        """Save a screenshot
        
        Args:
            frame: Frame to save
            directory (str): Directory to save to
            filename (str): Filename to save as
            
        Returns:
            str: Full path to saved screenshot
        """
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        screenshot_path = f"{directory}/{filename}"
        cv2.imwrite(screenshot_path, frame)
        return screenshot_path