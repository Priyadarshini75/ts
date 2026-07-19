"""
Configuration for Jewelry Theft Detection System.
All thresholds and settings are centralized here for easy tuning.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionConfig:
    """Main configuration for the theft detection pipeline."""

    # ── YOLOv7 Pose Model ──────────────────────────────────────────
    pose_model: str = "weights/yolov7-w6-pose.pt"
    pose_confidence: float = 0.3
    pose_iou_threshold: float = 0.45
    img_size: int = 640

    # ── ByteTrack Settings ──────────────────────────────────────────
    track_thresh: float = 0.4
    track_buffer: int = 30
    match_thresh: float = 0.8

    # ── Video Processing ────────────────────────────────────────────
    process_fps: int = 10
    output_fps: int = 10
    output_codec: str = "mp4v"

    # ── COCO Keypoint Indices ────────────────────────────────────────
    KP_NOSE: int = 0
    KP_LEFT_EYE: int = 1
    KP_RIGHT_EYE: int = 2
    KP_LEFT_EAR: int = 3
    KP_RIGHT_EAR: int = 4
    KP_LEFT_SHOULDER: int = 5
    KP_RIGHT_SHOULDER: int = 6
    KP_LEFT_ELBOW: int = 7
    KP_RIGHT_ELBOW: int = 8
    KP_LEFT_WRIST: int = 9
    KP_RIGHT_WRIST: int = 10
    KP_LEFT_HIP: int = 11
    KP_RIGHT_HIP: int = 12
    KP_LEFT_KNEE: int = 13
    KP_RIGHT_KNEE: int = 14
    KP_LEFT_ANKLE: int = 15
    KP_RIGHT_ANKLE: int = 16

    # ── Hand-to-Mouth Detection ─────────────────────────────────────
    hand_mouth_ratio: float = 0.6      # wrist-to-nose / shoulder_width
    hand_mouth_min_frames: int = 5
    hand_mouth_score: float = 40.0

    # ── Hand-to-Pocket Detection ────────────────────────────────────
    hand_pocket_ratio: float = 0.5     # wrist-to-hip / shoulder_width
    hand_pocket_min_frames: int = 5
    hand_pocket_score: float = 35.0

    # ── Head Movement (Looking Around) ──────────────────────────────
    head_movement_window: int = 20     # frames in sliding window
    head_direction_changes: int = 5    # min reversals to flag (raised from 3)
    head_movement_score: float = 30.0

    # ── FER Face Expression ─────────────────────────────────────────
    fer_window: int = 10               # frames to average emotion scores
    fer_fear_threshold: float = 0.35   # avg fear score to trigger
    fer_surprise_threshold: float = 0.50
    fer_fear_score: float = 25.0
    fer_surprise_score: float = 15.0
    fer_min_face_size: int = 30        # min pixels to run FER (skip tiny faces)
    fer_kp_conf_min: float = 0.5       # min keypoint conf to crop face

    # ── Anomaly Score System ────────────────────────────────────────
    alert_threshold: float = 65.0      # WARNING
    high_alert_threshold: float = 80.0 # HIGH
    score_decay: float = 0.95          # per-frame score decay
    temporal_bonus: float = 1.3        # bonus for concealment + post-theft combo
    alert_cooldown_sec: float = 10.0   # min seconds between alerts per person

    # ── Visualization Colors (BGR) ──────────────────────────────────
    color_normal: Tuple[int, int, int] = (0, 255, 0)
    color_warning: Tuple[int, int, int] = (0, 165, 255)
    color_alert: Tuple[int, int, int] = (0, 0, 255)
    color_skeleton: Tuple[int, int, int] = (255, 255, 0)
