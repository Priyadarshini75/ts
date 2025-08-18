import os
import cv2
import torch
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
from model_loader import get_yoloe_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FireSmokeDetection:
    def __init__(self, confidence_threshold=0.5):
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing FireSmokeDetection on device: {self.device}")
        
        try:
            self.model = get_yoloe_model()
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

        # Target classes for detection (should match config.py)
        self.target_classes = ["fire", "spark"]
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
        self.scales = [1.0]  # Start with 1.0 scale, can add more if needed
        self.history_threshold = 5  # Number of frames to confirm detection

        self.processed_track_ids = set()
        self.detection_history = {}

        # Screenshot directory
        self.screenshot_dir = "media/fire_spark"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        logger.info(f"Screenshots will be saved to: {os.path.abspath(self.screenshot_dir)}")

    def save_screenshot(self, frame, track_id, label, confidence):
        """Save screenshot with detection info"""
        try:
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{label}_track{track_id}_{timestamp_str}.jpg"
            path = os.path.join(self.screenshot_dir, filename)

            # Create a copy of the frame to draw on
            frame_copy = frame.copy()
            
            # Draw detection info
            overlay_text = f"{label.upper()} (Conf: {confidence:.2f})"
            cv2.putText(frame_copy, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            
            # Save the image
            cv2.imwrite(path, frame_copy)
            logger.info(f"Screenshot saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving screenshot: {str(e)}")
            return None

    def _check_detection_history(self, track_id, label, confidence):
        """Track fire/spark detection history"""
        if track_id not in self.detection_history:
            self.detection_history[track_id] = {
                "label": label,
                "frames": 0,
                "confidence_sum": 0.0,
                "finalized": False
            }

        self.detection_history[track_id]["frames"] += 1
        self.detection_history[track_id]["confidence_sum"] += confidence

        if self.detection_history[track_id]["frames"] >= self.history_threshold:
            if not self.detection_history[track_id]["finalized"]:
                avg_conf = self.detection_history[track_id]["confidence_sum"] / self.history_threshold
                self.detection_history[track_id]["confidence"] = avg_conf
                self.detection_history[track_id]["finalized"] = True
                return True, avg_conf

        return False, confidence

    async def run(self, payload, selected_use_key="fire_spark_detection", display=False, max_frames=None, event_service=None):
        logger.info(f"Starting fire/spark detection with payload: {payload}")
        camera_url = payload["camera_detail"]["source_url"]
        
        # Validate input source
        if not os.path.exists(camera_url) and not camera_url.startswith(('rtsp://', 'http://', 'https://')):
            logger.error(f"Invalid video source: {camera_url}")
            return

        use_case = next((uc for uc in payload["use_cases"] if uc["use_key"] == selected_use_key), None)
        if not use_case:
            logger.error(f"Use case '{selected_use_key}' not found in payload.")
            raise ValueError(f"Use case '{selected_use_key}' not found in payload.")

        # Set up video capture
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {camera_url}")
            return

        frame_count = 0
        logger.info("Starting detection loop...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame, retrying...")
                    await asyncio.sleep(1)
                    continue

                frame_count += 1
                detections_this_frame = []
                
                # Process frame at different scales
                for scale in self.scales:
                    try:
                        # Resize frame if scale is not 1.0
                        resized_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame
                        
                        # Run inference
                        with torch.no_grad():
                            results = self.model(resized_frame)[0]
                        
                        # Process detections
                        for det in results.boxes.data.tolist():
                            x1, y1, x2, y2, conf, cls_id = det[:6]  # Ensure we only take first 6 elements
                            label = results.names[int(cls_id)]
                            
                            # Scale coordinates back to original frame size if we resized
                            if scale != 1.0:
                                x1, y1 = int(x1 / scale), int(y1 / scale)
                                x2, y2 = int(x2 / scale), int(y2 / scale)
                            else:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Check if detection is a fire or spark
                            if conf >= self.confidence_threshold and any(c in label.lower() for c in self.target_classes):
                                detections_this_frame.append((x1, y1, x2, y2, label, conf))
                                logger.debug(f"Detected {label} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                    
                    except Exception as e:
                        logger.error(f"Error processing frame at scale {scale}: {str(e)}")
                        continue

                # Process detections
                for track_id, det in enumerate(detections_this_frame):
                    x1, y1, x2, y2, label, conf = det
                    
                    # Draw detection on frame for display
                    if display:
                        color = (0, 0, 255) if "fire" in label.lower() else (0, 255, 255)  # Red for fire, Yellow for smoke
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Check detection history and trigger events if needed
                    finalized, avg_conf = self._check_detection_history(track_id, label, conf)
                    
                    if finalized and track_id not in self.processed_track_ids:
                        self.processed_track_ids.add(track_id)
                        screenshot_path = self.save_screenshot(frame.copy(), track_id, label, avg_conf)
                        
                        if event_service:
                            event_data = {
                                "headers": ["track_id", "label", "confidence", "status"],
                                "values": [str(track_id), label, f"{avg_conf:.2f}", "Detected"],
                                "labels": ["Track ID", "Type", "Confidence", "Status"],
                            }

                            event_repo = {
                                "task_id": payload["task_id"],
                                "camera_id": payload["camera_id"],
                                "use_case_id": use_case["use_case_id"],
                                "use_key": use_case["use_key"],
                                "event_type": f"{label}_detected",
                                "severity": "HIGH" if "fire" in label.lower() else "MEDIUM",
                                "message": f"{label} detected with confidence {avg_conf:.2f}",
                                "event_data": event_data,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "track_id": track_id,
                                "screenshot_path": screenshot_path
                            }
                            
                            try:
                                await event_service.create_event(event_data=event_repo)
                                logger.info(f"Event created for {label} detection (Track ID: {track_id})")
                            except Exception as e:
                                logger.error(f"Failed to create event: {str(e)}")

                # Display frame if enabled
                if display:
                    cv2.imshow("Fire/Spark Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested to stop")
                        break

                # Check frame limit
                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached maximum frame limit: {max_frames}")
                    break
                    
                # Small delay to prevent high CPU usage
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in detection loop: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            # Clean up
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info("Detection stopped")


