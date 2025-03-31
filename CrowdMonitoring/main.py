import cv2
from config.config import *
from utils.video_utils import initialize_video_writer, release_resources
from utils.screenshot_utils import save_screenshot
from utils.db_utils import get_mongo_collection, insert_crowd_data
from detector.yolo_detector import load_yolo_model, detect_people
from detector.grid_analysis import analyze_crowd

def main():
    # Initialize MongoDB collection
    collection = get_mongo_collection(MONGO_URI, MONGO_DB, MONGO_COLLECTION)

    # Initialize video
    cap, out, frame_width, frame_height = initialize_video_writer(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

    # Load YOLO model
    model = load_yolo_model(YOLO_MODEL_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people in the frame
        people_boxes = detect_people(model, frame)

        # Analyze crowd in the grid
        detected_crowds = analyze_crowd(frame, people_boxes, GRID_SIZE, frame_width, frame_height, CROWD_THRESHOLD, DISTANCE_THRESHOLD)

        # Save screenshots and insert into DB
        for _ in detected_crowds:
            screenshot_path = save_screenshot(frame, SCREENSHOT_FOLDER)
            insert_crowd_data(collection, screenshot_path)

        # Write frame to output video
        out.write(frame)

        # Display the frame
        cv2.imshow("Crowd Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_resources(cap, out)

if __name__ == "__main__":
    main()
