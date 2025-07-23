import cv2
from config import *
from model import get_model
from tracker import get_tracker
from preprocessor import prepare_detections
from util import save_alert

model = get_model()
tracker = get_tracker()

cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
person_bag_map = {}
unattended_bags = {}
bag_owner = {}
bag_alerted = set()
suspicious_alerted = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]
    detections = prepare_detections(results, model, CONF_THRESHOLD, PERSON_CONF_THRESHOLD)

    dets_for_tracker = [([x1, y1, x2 - x1, y2 - y1], score, cls_name)
                        for x1, y1, x2, y2, score, cls_name in detections]

    tracks = tracker.update_tracks(dets_for_tracker, frame=frame)
    current_people = {}
    current_bags = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cls = track.get_det_class()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if cls == "person":
            cv2.putText(frame, f"{cls} {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            current_people[track_id] = [x1, y1, x2, y2]
            person_bag_map[track_id] = {'last_seen': frame_count, 'box': [x1, y1, x2, y2]}
        elif "bag" in cls or "suitcase" in cls or "backpack" in cls or "handbag" in cls:
            current_bags[track_id] = [x1, y1, x2, y2]

    for bag_id, box in current_bags.items():
        bx1, by1, bx2, by2 = box
        assigned = False
        for pid, pdata in current_people.items():
            px1, py1, px2, py2 = pdata
            if px1 < bx1 < px2 and py1 < by1 < py2:
                bag_owner[bag_id] = pid
                assigned = True
                unattended_bags.pop(bag_id, None)
                break

        if not assigned:
            if bag_id not in unattended_bags:
                unattended_bags[bag_id] = {
                    'box': box,
                    'start_frame': frame_count,
                    'owner': bag_owner.get(bag_id)
                }
            elif frame_count - unattended_bags[bag_id]['start_frame'] > UNATTENDED_DELAY:
                if bag_id not in bag_alerted:
                    save_alert(frame, box, "Unattended", "unattended", UNATTENDED_DIR, bag_id)
                    bag_alerted.add(bag_id)

    for bag_id, info in unattended_bags.items():
        bx1, by1, bx2, by2 = info['box']
        for pid, pdata in current_people.items():
            if pid == info['owner']:
                continue
            px1, py1, px2, py2 = pdata
            if px1 < bx1 < px2 and py1 < by1 < py2:
                if bag_id not in suspicious_alerted:
                    save_alert(frame, info['box'], "Suspicious", "suspicious", SUSPICIOUS_DIR, bag_id)
                    suspicious_alerted.add(bag_id)

    out.write(frame)
    cv2.imshow("Unattended Bag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
