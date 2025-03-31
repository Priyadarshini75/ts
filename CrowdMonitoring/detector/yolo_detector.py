from ultralytics import YOLO

def load_yolo_model(model_path):
    return YOLO(model_path)

def detect_people(model, frame):
    results = model.track(frame, persist=True)
    people_boxes = []
    for result in results:
        if result.boxes is not None:
            for track in result.boxes:
                if track.cls == 0:  # Class ID 0 corresponds to persons in COCO dataset
                    x1, y1, x2, y2 = track.xyxy[0].cpu().numpy().astype(int)
                    track_id = track.id
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    people_boxes.append((center, track_id))
    return people_boxes
