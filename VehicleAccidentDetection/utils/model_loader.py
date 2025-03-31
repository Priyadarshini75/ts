from ultralytics import YOLO

class ModelLoader:
    def __init__(self, accident_model_path, vehicle_model_path):
        """Initialize and load YOLO models"""
        print(f"Loading accident detection model from: {accident_model_path}")
        self.accident_model = YOLO(accident_model_path)
        
        print(f"Loading vehicle detection model from: {vehicle_model_path}")
        self.vehicle_model = YOLO(vehicle_model_path)
        
        # Define class categories
        self.accident_classes = [
            "bike_bike_accident", "bike_object_accident", "bike_person_accident",
            "car_bike_accident", "car_car_accident", "car_object_accident", 
            "car_person_accident"
        ]
        
        self.vehicle_classes = ['person', 'truck', 'bus', 'car', 'motorcycle']
        
        # Enhanced color palette with softer tones
        self.colors = {
            'person': (90, 200, 140),     # Soft green
            'truck': (180, 130, 90),      # Muted brown
            'bus': (210, 160, 120),       # Warm tan
            'car': (140, 180, 210),       # Soft blue
            'motorcycle': (180, 120, 220) # Soft purple
        }
        
        print("Models loaded successfully")