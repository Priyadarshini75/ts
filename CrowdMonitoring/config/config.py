import os

# Configuration for paths
INPUT_VIDEO_PATH = os.path.join("data", "input_videos", "853889-hd_1920_1080_25fps.mp4")
OUTPUT_VIDEO_PATH = os.path.join("data", "output_videos", "output_video.avi")
SCREENSHOT_FOLDER = os.path.join("data", "screenshots")

# Create folders if not exist
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "crowd_monitoring_db"
MONGO_COLLECTION = "crowd_data"

# YOLO model configuration
YOLO_MODEL_PATH = "data\models\yolov8n.pt"

# Grid configuration
GRID_SIZE = (3, 3)

# Thresholds
CROWD_THRESHOLD = 10  # Minimum number of people to consider as a crowd
DISTANCE_THRESHOLD = 50
