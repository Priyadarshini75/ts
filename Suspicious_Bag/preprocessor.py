def prepare_detections(results, model, conf_thresh, person_conf_thresh):
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = r
        cls_name = model.names[int(cls_id)]
        if cls_name == "person" and score < person_conf_thresh:
            continue
        if score >= conf_thresh:
            detections.append((x1, y1, x2, y2, score, cls_name))
    return detections
