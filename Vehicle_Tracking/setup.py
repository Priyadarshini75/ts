# setup.py
import os
from config import MODEL_DIR

# Ensure necessary folders exist
os.makedirs("input_videos", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
