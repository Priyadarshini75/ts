# Configuration settings for the accident detection system

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to models
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONTAINER_MODEL_PATH = os.path.join(MODELS_DIR, "container_detection_model.pt")
TRUCK_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n.pt")

# Media directories
MEDIA_DIR = os.path.join(BASE_DIR, "media")
INPUT_DIR = os.path.join(MEDIA_DIR, "input")
OUTPUT_DIR = os.path.join(MEDIA_DIR, "output")
SCREENSHOTS_DIR = os.path.join(MEDIA_DIR, "screenshots")

# Ensure directories exist
for directory in [MODELS_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR, SCREENSHOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Input and output video paths
INPUT_VIDEO_PATH = os.path.join(INPUT_DIR, "Truck lifted up at container dock.mp4")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "output_video_with_accident_detection.mp4")

# MongoDB settings
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "ContainerAccident"
MONGO_COLLECTION = "Records"

# Detection parameters
MAX_HISTORY = 20
UPWARD_THRESHOLD = 15
DOWNWARD_THRESHOLD = 20
LIFT_HEIGHT_THRESHOLD = 30
MOVEMENT_THRESHOLD = 8