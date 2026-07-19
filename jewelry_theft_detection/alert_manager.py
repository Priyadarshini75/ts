"""
Alert Manager — generates alerts and saves annotated JPEG snapshots.
No video clips saved. One alert image per alert event.
"""

import cv2
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from .config import DetectionConfig
from .behavior_analyzer import BehaviorState


class AlertRecord:
    """Represents a single alert event."""

    def __init__(
        self,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        anomaly_score: float,
        behaviors: List[str],
        alert_level: str,
        image_path: Optional[str] = None,
    ):
        self.track_id = track_id
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.anomaly_score = anomaly_score
        self.behaviors = behaviors
        self.alert_level = alert_level
        self.image_path = image_path

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp, 2),
            "anomaly_score": round(self.anomaly_score, 1),
            "behaviors": self.behaviors,
            "alert_level": self.alert_level,
            "image_path": self.image_path,
        }


class AlertManager:
    """
    Manages alert generation.
    On alert trigger: saves one annotated JPEG image and logs to JSON.
    """

    def __init__(
        self,
        config: DetectionConfig,
        output_dir: str,
        fps: float,
    ):
        self.config = config
        self.output_dir = output_dir
        self.fps = fps

        os.makedirs(output_dir, exist_ok=True)

        self.alerts: List[AlertRecord] = []
        self.last_alert_frame: Dict[int, int] = {}
        self.alert_count: int = 0
        self.log_path = os.path.join(output_dir, "alerts_log.json")

    def check_and_alert(
        self,
        track_id: int,
        state: BehaviorState,
        frame_idx: int,
        annotated_frame: np.ndarray,
    ) -> Optional[AlertRecord]:
        """
        Check if this person should trigger an alert.
        If yes, save an annotated JPEG snapshot and log the event.

        Args:
            track_id:        ByteTrack person ID
            state:           BehaviorState for this person
            frame_idx:       Current processed frame index
            annotated_frame: Current annotated BGR frame (for saving image)

        Returns:
            AlertRecord if alert triggered, else None
        """
        alert_level = state.get_alert_level()
        if alert_level == "NORMAL":
            return None

        cooldown_frames = int(self.config.alert_cooldown_sec * self.fps)
        last_frame = self.last_alert_frame.get(track_id, -9999)
        if frame_idx - last_frame < cooldown_frames:
            return None

        timestamp = frame_idx / self.fps
        self.alert_count += 1

        # Save alert image
        image_path = self._save_alert_image(
            annotated_frame, track_id, frame_idx, alert_level
        )

        alert = AlertRecord(
            track_id=track_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            anomaly_score=state.anomaly_score,
            behaviors=list(state.active_behaviors),
            alert_level=alert_level,
            image_path=image_path,
        )

        self.alerts.append(alert)
        self.last_alert_frame[track_id] = frame_idx

        # ── Console output ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"🚨 ALERT #{self.alert_count} — {alert_level}")
        print(f"   Person ID:  {track_id}")
        print(f"   Time:       {int(timestamp//60):02d}:{int(timestamp%60):02d}")
        print(f"   Score:      {state.anomaly_score:.1f}/100")
        print(f"   Behaviors:  {', '.join(state.active_behaviors)}")
        if image_path:
            print(f"   📸 Saved:   {image_path}")
        print(f"{'='*60}\n")

        return alert

    def _save_alert_image(
        self,
        frame: np.ndarray,
        track_id: int,
        frame_idx: int,
        alert_level: str,
    ) -> str:
        """
        Save the current annotated frame as a JPEG alert snapshot.
        Adds a bold colored banner at the top indicating the alert.
        """
        img = frame.copy()
        h, w = img.shape[:2]

        # ── Banner color by level ──────────────────────────────────
        if alert_level == "HIGH":
            banner_color = (0, 0, 200)       # deep red
            text_color = (255, 255, 255)
        else:
            banner_color = (0, 120, 220)     # orange
            text_color = (255, 255, 255)

        banner_h = 50
        cv2.rectangle(img, (0, 0), (w, banner_h), banner_color, -1)

        label = (
            f"ALERT #{self.alert_count} | Person {track_id} | "
            f"Frame {frame_idx} | {alert_level}"
        )
        cv2.putText(
            img, label, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA
        )

        filename = (
            f"alert_{self.alert_count:04d}_"
            f"person{track_id}_"
            f"frame{frame_idx}_"
            f"{alert_level}.jpg"
        )
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    def save_alert_log(self):
        """Write all alerts to alerts_log.json."""
        log_data = {
            "total_alerts": len(self.alerts),
            "generated_at": datetime.now().isoformat(),
            "alerts": [a.to_dict() for a in self.alerts],
        }
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"\n📋 Alert log saved: {self.log_path}")
        print(f"   Total alerts: {len(self.alerts)}")
