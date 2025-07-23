# init.py
import os
import urllib.request
from config import MODEL_DIR, MODEL_PATH, MODEL_URL

# Create necessary directories
os.makedirs("input_videos", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print(f"[INFO] Downloading model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded successfully.")
else:
    print("[INFO] Model already exists.")
