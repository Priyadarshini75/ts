"""
FER Face Expression Detector.
Uses the FER library (Mini-Xception model) to detect fear/anxiety
from face regions cropped using YOLOv7-Pose keypoints.
"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("[FaceExpression] WARNING: FER library not installed. "
          "Run: pip install fer tensorflow")

from .config import DetectionConfig
from .pose_detector import PersonPose


class FaceExpressionDetector:
    """
    Detects fear/anxiety emotions from face crops using FER.
    Uses YOLOv7-Pose keypoints to locate the face region
    — no separate face detector needed.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.available = FER_AVAILABLE

        if self.available:
            # mtcnn=False: use faster OpenCV face detection inside FER
            # (we already have face location from keypoints)
            self.detector = FER(mtcnn=False)
            print("[FaceExpression] FER detector initialized (Mini-Xception model).")
        else:
            self.detector = None
            print("[FaceExpression] FER not available — face expression disabled.")

        # Per-track emotion history for temporal smoothing
        # track_id -> deque of (fear, surprise) tuples
        self.emotion_history: Dict[int, deque] = {}

    def _get_face_crop(self, frame: np.ndarray, person: PersonPose) -> Optional[np.ndarray]:
        """
        Crop the face region from frame using pose keypoints.
        Uses nose, eyes, ears (kp 0-4) to build face bounding box.

        Returns:
            Cropped face image or None if keypoints not reliable enough.
        """
        cfg = self.config
        h, w = frame.shape[:2]

        # Gather visible face keypoints (nose, eyes, ears)
        face_kp_indices = [cfg.KP_NOSE, cfg.KP_LEFT_EYE, cfg.KP_RIGHT_EYE,
                           cfg.KP_LEFT_EAR, cfg.KP_RIGHT_EAR]

        visible_points = []
        for idx in face_kp_indices:
            if person.kp_conf[idx] >= cfg.fer_kp_conf_min:
                visible_points.append(person.keypoints[idx])

        # Need at least 2 face keypoints
        if len(visible_points) < 2:
            return None

        visible_points = np.array(visible_points)
        x_min = int(np.min(visible_points[:, 0]))
        y_min = int(np.min(visible_points[:, 1]))
        x_max = int(np.max(visible_points[:, 0]))
        y_max = int(np.max(visible_points[:, 1]))

        # Add 40% padding around the keypoint bounding box
        pad_x = max(int((x_max - x_min) * 0.4), 15)
        pad_y = max(int((y_max - y_min) * 0.4), 15)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        # Skip if face crop is too small
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        if crop_w < cfg.fer_min_face_size or crop_h < cfg.fer_min_face_size:
            return None

        return frame[y_min:y_max, x_min:x_max]

    def detect(self, frame: np.ndarray, track_id: int,
               person: PersonPose) -> Tuple[float, float]:
        """
        Detect fear and surprise emotion levels for a tracked person.
        Applies temporal smoothing over the last N frames.

        Args:
            frame: Full BGR video frame
            track_id: Person's track ID
            person: Current pose detection

        Returns:
            (avg_fear, avg_surprise) — smoothed over fer_window frames
        """
        if not self.available:
            return 0.0, 0.0

        # Initialize history for new tracks
        if track_id not in self.emotion_history:
            self.emotion_history[track_id] = deque(
                maxlen=self.config.fer_window
            )

        # Get face crop
        face_crop = self._get_face_crop(frame, person)

        if face_crop is not None:
            try:
                # FER expects BGR image
                emotions = self.detector.detect_emotions(face_crop)

                if emotions:
                    # Take highest confidence face detection
                    scores = emotions[0]['emotions']
                    fear = float(scores.get('fear', 0.0))
                    surprise = float(scores.get('surprise', 0.0))
                    self.emotion_history[track_id].append((fear, surprise))
                else:
                    # No face detected in crop — append zeros
                    self.emotion_history[track_id].append((0.0, 0.0))
            except Exception:
                self.emotion_history[track_id].append((0.0, 0.0))
        else:
            # Can't crop face — append zeros
            self.emotion_history[track_id].append((0.0, 0.0))

        # Return smoothed averages
        history = list(self.emotion_history[track_id])
        if not history:
            return 0.0, 0.0

        avg_fear = float(np.mean([h[0] for h in history]))
        avg_surprise = float(np.mean([h[1] for h in history]))

        return avg_fear, avg_surprise

    def cleanup_lost_tracks(self, active_track_ids: set):
        """Remove emotion history for persons no longer tracked."""
        lost = [tid for tid in self.emotion_history if tid not in active_track_ids]
        for tid in lost:
            del self.emotion_history[tid]
