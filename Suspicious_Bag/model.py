from ultralytics import YOLO
from config import CLASSES, MODEL_NAME, MODEL_FOLDER
import os

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
        _model = YOLO(model_path)
        text_pe = _model.get_text_pe(CLASSES)
        _model.set_classes(CLASSES, text_pe)
    return _model
