# model.py
from ultralytics import YOLOE
from config import MODEL_PATH, CLASSES

def load_model():
    model = YOLOE(MODEL_PATH)  # Will auto-download to ultralytics cache if not found
    text_pe = model.get_text_pe(CLASSES)
    model.set_classes(CLASSES, text_pe)
    return model
