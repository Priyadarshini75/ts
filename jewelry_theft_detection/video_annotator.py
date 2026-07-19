"""
Video Annotator - Draws pose overlays and alert information on output video frames.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from .config import DetectionConfig
from .pose_detector import PersonPose
from .behavior_analyzer import BehaviorState


# Connections for drawing skeletons
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]

BEHAVIOR_LABELS = {
    "HAND_TO_MOUTH":    "Hand to Mouth",
    "HAND_TO_POCKET":   "Hand to Pocket/Pants",
    "LOOKING_AROUND":   "Looking Around",
    "FEAR_DETECTED":    "😨 Fear Detected",
    "SURPRISE_DETECTED": "😲 Surprise Detected",
    "SEQUENCE_DETECTED": "⚠ SUSPICIOUS SEQUENCE",
}


class VideoAnnotator:
    """Draws pose skeletons, behavior labels, and alert overlays on video frames."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def annotate_frame(self, frame: np.ndarray, tracked_persons: Dict[int, PersonPose],
                       behavior_states: Dict[int, BehaviorState],
                       frame_idx: int, fps: float) -> np.ndarray:
        """Draw annotations on a frame."""
        annotated = frame.copy()

        for track_id, person in tracked_persons.items():
            state = behavior_states.get(track_id)
            alert_level = state.get_alert_level() if state else "NORMAL"

            if alert_level == "HIGH":
                color = self.config.color_alert
            elif alert_level == "WARNING":
                color = self.config.color_warning
            else:
                color = self.config.color_normal

            self._draw_skeleton(annotated, person, color)
            self._draw_bbox(annotated, person, track_id, color, alert_level)

            if state and state.active_behaviors:
                self._draw_behaviors(annotated, person, state)

            if state:
                self._draw_score_bar(annotated, person, state)

        self._draw_frame_info(annotated, frame_idx, fps, len(tracked_persons))
        return annotated

    def _draw_skeleton(self, frame: np.ndarray, person: PersonPose,
                       color: Tuple[int, int, int]):
        """Draw skeleton connections."""
        kps = person.keypoints
        confs = person.kp_conf

        for (i, j) in SKELETON_CONNECTIONS:
            if confs[i] > 0.3 and confs[j] > 0.3:
                pt1 = (int(kps[i][0]), int(kps[i][1]))
                pt2 = (int(kps[j][0]), int(kps[j][1]))
                cv2.line(frame, pt1, pt2, self.config.color_skeleton, 2)

        for idx in range(17):
            if confs[idx] > 0.3:
                pt = (int(kps[idx][0]), int(kps[idx][1]))
                cv2.circle(frame, pt, 4, color, -1)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1)

    def _draw_bbox(self, frame: np.ndarray, person: PersonPose,
                   track_id: int, color: Tuple[int, int, int], alert_level: str):
        """Draw bounding box."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        thickness = 3 if alert_level != "NORMAL" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{track_id}"
        if alert_level != "NORMAL":
            label += f" [{alert_level}]"

        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - baseline - 8),
            (x1 + label_w + 8, y1),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x1 + 4, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    def _draw_behaviors(self, frame: np.ndarray, person: PersonPose,
                        state: BehaviorState):
        """Draw active behaviors next to person."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        y_offset = y2 + 20

        for behavior in state.active_behaviors:
            label = BEHAVIOR_LABELS.get(behavior, behavior)

            if behavior in ("HAND_TO_MOUTH", "HAND_TO_POCKET"):
                bg_color = (0, 0, 180)        # dark red — concealment
            elif behavior == "SEQUENCE_DETECTED":
                bg_color = (0, 0, 255)        # bright red — high risk
            elif behavior in ("FEAR_DETECTED", "SURPRISE_DETECTED"):
                bg_color = (150, 0, 200)      # purple — emotion signal
            else:
                bg_color = (0, 120, 200)      # blue — general

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y_offset - th - 4),
                (x1 + tw + 8, y_offset + 4),
                bg_color, -1
            )
            cv2.putText(
                frame, label,
                (x1 + 4, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += th + 12

    def _draw_score_bar(self, frame: np.ndarray, person: PersonPose,
                        state: BehaviorState):
        """Draw suspicion score bar."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]

        bar_width = x2 - x1
        bar_height = 8
        bar_y = y1 - 12

        score = state.anomaly_score
        fill_width = int(bar_width * min(score / 100.0, 1.0))

        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), -1)

        if score >= 80:
            fill_color = (0, 0, 255)      # red — HIGH
        elif score >= 65:
            fill_color = (0, 165, 255)    # orange — WARNING
        elif score >= 30:
            fill_color = (0, 255, 255)    # yellow — elevated
        else:
            fill_color = (0, 255, 0)      # green — normal

        if fill_width > 0:
            cv2.rectangle(
                frame, (x1, bar_y), (x1 + fill_width, bar_y + bar_height),
                fill_color, -1
            )

        score_text = f"{score:.0f}%"
        cv2.putText(
            frame, score_text,
            (x2 + 5, bar_y + bar_height),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

    def _draw_frame_info(self, frame: np.ndarray, frame_idx: int,
                         fps: float, num_persons: int):
        """Draw frame overlay."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        time_sec = frame_idx / fps if fps > 0 else 0
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)

        info_lines = [
            f"Frame: {frame_idx}  |  Time: {minutes:02d}:{seconds:02d}",
            f"Persons Detected: {num_persons}",
            f"JEWELRY THEFT DETECTION SYSTEM",
        ]

        y = 30
        for line in info_lines:
            cv2.putText(
                frame, line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1
            )
            y += 25
