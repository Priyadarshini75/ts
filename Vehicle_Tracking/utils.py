# utils.py
import cv2

def draw_boxes_and_count(frame, boxes, classes):
    count = 0
    for box in boxes:
        x1, y1, x2, y2, score, cls_id = box
        cls_id = int(cls_id)
        cls_name = classes[cls_id] if cls_id < len(classes) else "object"

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{cls_name} {score:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        count += 1

    cv2.putText(frame, f"total_vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return frame
