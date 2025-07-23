import cv2
import os
from datetime import datetime
from config import UNATTENDED_DIR, SUSPICIOUS_DIR

def save_alert(frame, box, label, tag, folder, bag_id):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{tag}_bag_{bag_id}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
