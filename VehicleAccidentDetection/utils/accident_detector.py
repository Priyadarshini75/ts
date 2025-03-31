class AccidentDetector:
    def __init__(self, model_loader):
        """Initialize accident detector with loaded models"""
        self.model_loader = model_loader
    
    def detect_accident(self, frame, confidence_threshold=0.7):
        """
        Perform accident detection on a frame
        
        Args:
            frame: The video frame to analyze
            confidence_threshold: Minimum confidence score to consider a detection valid
            
        Returns:
            Boolean indicating if an accident was detected
        """
        results = self.model_loader.accident_model(frame)
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0]
            class_name = self.model_loader.accident_model.names[cls]
            
            if (class_name in self.model_loader.accident_classes and 
                confidence > confidence_threshold):
                return True
        return False
    
    def detect_vehicles(self, frame):
        """
        Perform vehicle detection on a frame
        
        Args:
            frame: The video frame to analyze
            
        Returns:
            Detection results from the vehicle model
        """
        return self.model_loader.vehicle_model(frame)