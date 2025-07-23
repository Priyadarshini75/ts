# main.py
import setup  # Ensures folders exist

from config import INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, CLASSES, CONF_THRESHOLD
from preprocessor import initialize_video_io
from model import load_model
from utils import draw_boxes_and_count
import cv2

def run():
    model = load_model()
    cap, out = initialize_video_io(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD)[0]
        boxes = results.boxes.data.tolist() if results.boxes is not None else []

        frame = draw_boxes_and_count(frame, boxes, CLASSES)

        out.write(frame)
        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
