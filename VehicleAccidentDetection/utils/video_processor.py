import cv2
import os
from datetime import datetime

class VideoProcessor:
    def __init__(self, input_path, output_path, detector, 
                 screenshot_dir='screenshots', screenshot_cooldown=2):
        """
        Initialize video processor
        
        Args:
            input_path: Path to input video file
            output_path: Path to save processed video
            detector: AccidentDetector instance
            screenshot_dir: Directory to save accident screenshots
            screenshot_cooldown: Cooldown between screenshots in seconds
        """
        self.input_path = input_path
        self.output_path = output_path
        self.detector = detector
        self.screenshot_dir = screenshot_dir
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Open input video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open input video: {input_path}")
        
        # Get video information
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video: {input_path}")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                  (self.frame_width, self.frame_height))
        
        # Configure screenshot settings
        self.screenshot_cooldown = screenshot_cooldown  # seconds between screenshots
    
    def annotate_frame(self, frame, accident_detected, vehicle_results):
        """
        Annotate frame with detection results
        
        Args:
            frame: Original video frame
            accident_detected: Boolean indicating if accident was detected
            vehicle_results: Results from vehicle detector
            
        Returns:
            Annotated frame with bounding boxes and labels
        """
        annotated_frame = frame.copy()
        
        # Add accident warning
        if accident_detected:
            cv2.rectangle(annotated_frame, (0, 0), (self.frame_width, 60), (50, 50, 255), -1)
            cv2.putText(annotated_frame, "Accident Detected!", (10, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 3)
        
        # Draw bounding boxes for vehicles and people
        for result in vehicle_results:
            boxes = result.boxes
            for box in boxes:
                # Get details
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.detector.model_loader.vehicle_model.names[cls]
                
                # Refined confidence check for trucks and buses
                if class_name in ['truck', 'bus'] and conf < 0.7:
                    continue
                
                # Filter specific classes
                if class_name in self.detector.model_loader.vehicle_classes:
                    # Select color
                    color = self.detector.model_loader.colors.get(class_name, (255, 255, 255))
                    
                    # Rounded rectangle for bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Stylish label with semi-transparent background
                    label = f"{class_name} {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1-h-10), (x1+w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def save_screenshot(self, frame, frame_number):
        """
        Save screenshot of detected accident
        
        Args:
            frame: Frame to save
            frame_number: Current frame number
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/accident_{current_time}_frame{frame_number}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def process(self):
        """Process the entire video"""
        frame_number = 0
        last_accident_frame = -1  # Track the last frame where an accident was detected
        
        print(f"Starting video processing...")
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Progress update every 100 frames
                if frame_number % 100 == 0:
                    progress = (frame_number / self.total_frames) * 100
                    print(f"Processing: {progress:.1f}% complete (frame {frame_number}/{self.total_frames})")
                
                # Perform inference
                accident_detected = self.detector.detect_accident(frame)
                vehicle_results = self.detector.detect_vehicles(frame)
                
                # Annotate frame
                annotated_frame = self.annotate_frame(frame, accident_detected, vehicle_results)
                
                # Save screenshot if accident detected (with cooldown to avoid saving too many)
                cooldown_frames = int(self.fps * self.screenshot_cooldown)
                if accident_detected and (frame_number - last_accident_frame > cooldown_frames):
                    self.save_screenshot(annotated_frame, frame_number)
                    last_accident_frame = frame_number
                
                # Write frame to output video
                self.out.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Accident Detection', annotated_frame)
                
                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing stopped by user")
                    break
                
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Cleanup
            self.cap.release()
            self.out.release()
            cv2.destroyAllWindows()
            
            print(f"Processed video saved at: {self.output_path}")
            print(f"Accident screenshots saved in: {self.screenshot_dir}")