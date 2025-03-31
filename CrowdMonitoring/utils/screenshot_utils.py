import cv2
import os
from datetime import datetime

def save_screenshot(frame, screenshot_folder):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = os.path.join(screenshot_folder, f"crowd_detected_{timestamp}.png")
    cv2.imwrite(screenshot_path, frame)
    return screenshot_path
