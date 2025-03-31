import cv2
import numpy as np
from utils.db_handler import MongoDBHandler
from utils.person_tracking import PersonTracker
import time

def main():
    # RTSP configuration
    rtsp_link = "rtsp://<your-rtsp-link>"
    camera_name = "Camera_1"

    # MongoDB setup
    db_handler = MongoDBHandler(
        db_name="PersonTrack", 
        collection_name="Records", 
        camera_name=camera_name,
        video_source=rtsp_link
    )

    # Camera source setup
    camera_source = rtsp_link if cv2.VideoCapture(rtsp_link).isOpened() else 0  # Fallback to webcam

    if camera_source == 0:
        print("RTSP link not available. Falling back to webcam.")

    # PersonTracker setup
    tracker = PersonTracker(
        yolo_model_path="models/yolo11n.pt",
        face_model_name='buffalo_l',
        mongo_handler=db_handler,
        use_case="Enhanced Person Tracking",
        camera_source=camera_source
    )

    # Video capture
    cap = cv2.VideoCapture(camera_source)
    
    # Tracking variables
    last_saved_count = 0
    save_interval = 5  # Save every 5 seconds
    last_save_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for person detection
        tracker.process_frame(frame)

        # Get total persons
        total_persons = len(tracker.person_database)

        # Display person count on frame
        cv2.putText(frame, f'Total Count: {total_persons}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show the processed frame
        cv2.imshow("Person Tracking", frame)

        # Real-time database updates with interval and count change
        current_time = time.time()
        if (current_time - last_save_time >= save_interval) or (total_persons != last_saved_count):
            # Insert event to MongoDB
            db_handler.insert_person_event(total_persons)
            
            # Update tracking variables
            last_saved_count = total_persons
            last_save_time = current_time

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()