from ultralytics import YOLO

class DetectionModels:
    """Class to handle loading and using detection models"""
    
    def __init__(self, container_model_path, truck_model_path):
        """Initialize the detection models
        
        Args:
            container_model_path (str): Path to the container detection model
            truck_model_path (str): Path to the truck detection model
        """
        self.container_model = YOLO(container_model_path)
        self.truck_model = YOLO(truck_model_path)
    
    def detect_containers(self, frame):
        """Detect containers in the frame
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            list: List of container bounding boxes
        """
        container_results = self.container_model(frame)
        container_boxes = [box.xyxy[0].tolist() for box in container_results[0].boxes]
        return container_boxes
    
    def detect_trucks(self, frame):
        """Detect trucks in the frame
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            list: List of truck bounding boxes
        """
        truck_results = self.truck_model(frame)
        truck_boxes = [box.xyxy[0].tolist() for box in truck_results[0].boxes]
        return truck_boxes