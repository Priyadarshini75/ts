import time
import os
from config.settings import *
from models.detection_models import DetectionModels
from utils.detection_utils import AccidentDetector
from utils.video_utils import VideoProcessor
from data.database import Database

def main():
    """Main function to run the accident detection system"""
    
    # Initialize components
    detection_models = DetectionModels(CONTAINER_MODEL_PATH, TRUCK_MODEL_PATH)
    video_processor = VideoProcessor(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    accident_detector = AccidentDetector(
        MAX_HISTORY, 
        MOVEMENT_THRESHOLD, 
        UPWARD_THRESHOLD, 
        DOWNWARD_THRESHOLD, 
        LIFT_HEIGHT_THRESHOLD
    )
    database = Database(MONGO_HOST, MONGO_PORT, MONGO_DB_NAME, MONGO_COLLECTION)
    
    # Set up video
    if not video_processor.setup_video():
        return
    
    # Process video frames
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = video_processor.read_frame()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect containers and trucks
        container_boxes = detection_models.detect_containers(frame)
        truck_boxes = detection_models.detect_trucks(frame)
        
        # Process truck detection
        current_truck_box = None
        if truck_boxes:
            # Use the largest truck box (likely the main truck in the scene)
            largest_area = 0
            for box in truck_boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > largest_area:
                    largest_area = area
                    current_truck_box = box
        
        # Update truck position and detect vertical movement
        current_truck_box, _ = accident_detector.update_truck_position(current_truck_box, frame_count)
        
        # Detect accident conditions
        accident_detected = accident_detector.detect_accident()
        
        # Draw detections
        frame = video_processor.draw_detections(frame, current_truck_box, container_boxes)
        
        # Handle accident detection
        if accident_detected:
            # Draw accident warning
            frame = video_processor.draw_accident_warning(frame)
            
            # Save screenshot and record to database (only once when accident is first detected)
            if not accident_detector.is_accident_recorded():
                timestamp = int(time.time())
                screenshot_filename = f"accident_{timestamp}.jpg"
                screenshot_path = video_processor.save_screenshot(
                    frame, SCREENSHOTS_DIR, screenshot_filename
                )
                print(f"Accident screenshot saved to: {screenshot_path}")
                
                # Save to MongoDB
                database.save_accident_record(screenshot_path)
                
                # Set flag to avoid multiple records of the same accident
                accident_detector.set_accident_recorded()
        
        # Show and write frame
        video_processor.show_frame(frame)
        video_processor.write_frame(frame)
        
        # Check for key press to exit
        if video_processor.check_key_press("q"):
            break
    
    # Release resources
    video_processor.release()
    print(f"Processed video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()