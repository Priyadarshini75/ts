import os

VIDEO_SOURCE = "IMG_1630.mov"
OUTPUT_PATH = "output_unattended_bag.mp4"
MODEL_NAME = "yoloe-11s-seg.pt"
MODEL_FOLDER = "models"

CONF_THRESHOLD = 0.5
PERSON_CONF_THRESHOLD = 0.7
UNATTENDED_DELAY = 50  # frames

CLASSES = ["person", "bag", "backpack", "handbag", "suitcase"]

UNATTENDED_DIR = "alerts/unattended_bag"
SUSPICIOUS_DIR = "alerts/suspicious_activity"
os.makedirs(UNATTENDED_DIR, exist_ok=True)
os.makedirs(SUSPICIOUS_DIR, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
