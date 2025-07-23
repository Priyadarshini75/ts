# config.py
import os

MODEL_NAME = "yoloe-11s-seg.pt"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

CLASSES = ["car", "bus", "truck", "motorcycle", "vehicle", "trucks", "bicycle", "van", "scooter", "trailer"]
CONF_THRESHOLD = 0.3

INPUT_VIDEO_PATH = "input_video/airport.mp4"
OUTPUT_VIDEO_PATH = "output_videos/output_vehicle_present_count.mp4"

