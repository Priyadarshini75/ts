class AccidentDetector:
    """Class to handle accident detection logic"""
    
    def __init__(self, max_history, movement_threshold, upward_threshold, downward_threshold, lift_height_threshold):
        """Initialize the accident detector
        
        Args:
            max_history (int): Maximum number of positions to keep in history
            movement_threshold (int): Threshold for detecting movement
            upward_threshold (int): Threshold for significant upward movement
            downward_threshold (int): Threshold for significant downward movement
            lift_height_threshold (int): Minimum height to consider as significant lift
        """
        self.truck_position_history = []
        self.truck_heights = []
        self.max_history = max_history
        self.movement_threshold = movement_threshold
        self.upward_threshold = upward_threshold
        self.downward_threshold = downward_threshold
        self.lift_height_threshold = lift_height_threshold
        
        # State tracking
        self.initial_truck_y = None
        self.max_lift_height = 0
        self.accident_detected = False
        self.accident_recorded = False
        self.truck_lifted = False
        self.truck_crashed = False
    
    def calculate_center(self, box):
        """Calculate the center point of a bounding box.
        
        Args:
            box (list): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            tuple: (x_center, y_center) coordinates
        """
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return (x_center, y_center)
    
    def detect_vertical_movement(self, position_history, frames=10):
        """Detect if an object is moving upward or downward based on its history.
        
        Args:
            position_history (list): List of position coordinates
            frames (int): Number of frames to consider for movement detection
            
        Returns:
            tuple: (movement_direction, movement_magnitude)
        """
        if len(position_history) < frames:
            return "stable", 0
        
        # Get y-coordinates of centers for the last few frames
        recent_y = [pos[1] for pos in position_history[-frames:]]
        
        # Calculate the overall vertical movement
        start_y = sum(recent_y[:3]) / 3  # Average of first 3 points to reduce noise
        end_y = sum(recent_y[-3:]) / 3   # Average of last 3 points to reduce noise
        
        movement = start_y - end_y  # Positive means upward movement (y decreases in image coordinates)
        
        if movement > self.movement_threshold:  # Threshold for significant upward movement
            return "upward", movement
        elif movement < -self.movement_threshold:  # Threshold for significant downward movement
            return "downward", -movement
        else:
            return "stable", 0

    def update_truck_position(self, truck_box, frame_count):
        """Update truck position history and check for vertical movement
        
        Args:
            truck_box (list): Truck bounding box coordinates
            frame_count (int): Current frame count
            
        Returns:
            tuple: Current truck position and center
        """
        if truck_box:
            truck_center = self.calculate_center(truck_box)
            self.truck_position_history.append(truck_center)
            
            # Record initial truck height for reference
            if self.initial_truck_y is None and frame_count > 10:  # Skip first few frames for stability
                self.initial_truck_y = truck_center[1]
            
            # Keep track of truck's vertical position
            if self.initial_truck_y is not None:
                # Positive value means truck is higher than initial position
                current_height = self.initial_truck_y - truck_center[1]
                self.truck_heights.append(current_height)
                
                # Update maximum lift height
                if current_height > self.max_lift_height:
                    self.max_lift_height = current_height
            
            # Limit history size
            if len(self.truck_position_history) > self.max_history:
                self.truck_position_history.pop(0)
            if len(self.truck_heights) > self.max_history:
                self.truck_heights.pop(0)
                
            return truck_box, truck_center
            
        return None, None
    
    def detect_accident(self):
        """Detect if an accident has occurred based on truck movement
        
        Returns:
            bool: True if accident is detected, False otherwise
        """
        if len(self.truck_position_history) > 10 and self.initial_truck_y is not None:
            # Get truck movement direction
            truck_movement, movement_magnitude = self.detect_vertical_movement(self.truck_position_history)
            
            # First phase: Detect if truck is being lifted
            if not self.truck_lifted and truck_movement == "upward" and movement_magnitude > self.upward_threshold:
                self.truck_lifted = True
            
            # Second phase: After lifting, detect if truck is crashing down
            if self.truck_lifted and not self.truck_crashed:
                # Check if truck is now moving downward rapidly
                if truck_movement == "downward" and movement_magnitude > self.downward_threshold:
                    # Also check if it was lifted significantly before falling
                    if self.max_lift_height > self.lift_height_threshold:
                        self.truck_crashed = True
                        self.accident_detected = True
        
        return self.accident_detected
    
    def is_accident_recorded(self):
        """Check if the accident has been recorded
        
        Returns:
            bool: True if accident has been recorded, False otherwise
        """
        return self.accident_recorded
    
    def set_accident_recorded(self):
        """Mark the accident as recorded"""
        self.accident_recorded = True