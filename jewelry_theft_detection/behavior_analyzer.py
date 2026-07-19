"""
Behavior Analyzer for Theft Detection.
Analyzes pose keypoints over time to detect suspicious behaviors:
  1. HAND_TO_MOUTH — concealing items in mouth
  2. HAND_TO_POCKET — hiding items in pants/pocket
  3. LOOKING_AROUND — nervous head scanning post-theft
  4. FEAR / SURPRISE — from FER face expression (passed in from main)

Removed: DUST_RUBBING, FIDGETING (unreliable in jewelry factory context).
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from .config import DetectionConfig
from .pose_detector import PersonPose


class BehaviorState:
    """Tracks behavioral signals and anomaly score for a single tracked person."""

    def __init__(self, config: DetectionConfig):
        self.config = config

        # ── Per-behavior frame counters ─────────────────────────────
        self.hand_mouth_counter: int = 0
        self.hand_pocket_counter: int = 0

        # ── Sliding window histories ────────────────────────────────
        self.nose_positions: deque = deque(maxlen=config.head_movement_window)
        self.shoulder_midpoints: deque = deque(maxlen=config.head_movement_window)

        # ── Anomaly score (0–100 scale) ─────────────────────────────
        self.anomaly_score: float = 0.0

        # ── Active behaviors this frame ─────────────────────────────
        self.active_behaviors: List[str] = []

        # ── History of (frame_idx, behaviors) for temporal sequencing
        self.behavior_history: deque = deque(maxlen=100)

        # ── Alert tracking ──────────────────────────────────────────
        self.last_alert_frame: int = -9999
        self.total_alerts: int = 0

    def get_alert_level(self) -> str:
        """Return alert level based on anomaly score."""
        if self.anomaly_score >= self.config.high_alert_threshold:
            return "HIGH"
        elif self.anomaly_score >= self.config.alert_threshold:
            return "WARNING"
        return "NORMAL"


class BehaviorAnalyzer:
    """
    Analyzes pose data and face emotions over time to produce
    a per-track anomaly score.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.person_states: Dict[int, BehaviorState] = defaultdict(
            lambda: BehaviorState(config)
        )

    def analyze(
        self,
        track_id: int,
        person: PersonPose,
        frame_idx: int,
        avg_fear: float = 0.0,
        avg_surprise: float = 0.0,
    ) -> BehaviorState:
        """
        Analyze a person's pose + face emotions for suspicious behavior.

        Args:
            track_id:    Persistent person ID from ByteTrack
            person:      Current frame's pose data
            frame_idx:   Current processed frame index
            avg_fear:    Smoothed fear score from FER (0.0 – 1.0)
            avg_surprise: Smoothed surprise score from FER (0.0 – 1.0)

        Returns:
            Updated BehaviorState for this person
        """
        state = self.person_states[track_id]
        state.active_behaviors = []

        # ── Get scale reference (shoulder width) ───────────────────
        shoulder_width = person.get_shoulder_width(self.config)
        if shoulder_width < 10:
            # Shoulders not visible → just decay and return
            state.anomaly_score *= self.config.score_decay
            return state

        # ── Run pose-based behavior detectors ─────────────────────
        self._check_hand_to_mouth(person, state, shoulder_width)
        self._check_hand_to_pocket(person, state, shoulder_width)
        self._check_head_movement(person, state, shoulder_width)

        # ── Integrate FER emotion scores ───────────────────────────
        self._integrate_face_emotion(state, avg_fear, avg_surprise)

        # ── Apply temporal sequence bonus ─────────────────────────
        self._apply_temporal_bonus(state)

        # ── Decay score (score fades if behavior stops) ───────────
        state.anomaly_score *= self.config.score_decay

        # ── Clamp to [0, 100] ─────────────────────────────────────
        state.anomaly_score = min(max(state.anomaly_score, 0.0), 100.0)

        # ── Record behavior history for temporal analysis ─────────
        if state.active_behaviors:
            state.behavior_history.append(
                (frame_idx, list(state.active_behaviors))
            )

        return state

    # ──────────────────────────────────────────────────────────────────
    # Pose-Based Detectors
    # ──────────────────────────────────────────────────────────────────

    def _check_hand_to_mouth(
        self, person: PersonPose, state: BehaviorState, shoulder_width: float
    ):
        """
        Detect hand moving to mouth/nose region.
        Workers don't naturally bring items to their mouth during factory work.
        """
        cfg = self.config
        nose = person.get_keypoint(cfg.KP_NOSE)
        left_wrist = person.get_keypoint(cfg.KP_LEFT_WRIST)
        right_wrist = person.get_keypoint(cfg.KP_RIGHT_WRIST)

        if nose is None:
            state.hand_mouth_counter = max(0, state.hand_mouth_counter - 1)
            return

        hand_near_mouth = False
        for wrist in [left_wrist, right_wrist]:
            if wrist is None:
                continue
            dist = np.linalg.norm(np.array(wrist) - np.array(nose))
            if dist / shoulder_width < cfg.hand_mouth_ratio:
                hand_near_mouth = True
                break

        if hand_near_mouth:
            state.hand_mouth_counter += 1
            if state.hand_mouth_counter >= cfg.hand_mouth_min_frames:
                state.active_behaviors.append("HAND_TO_MOUTH")
                state.anomaly_score += cfg.hand_mouth_score * 0.3
        else:
            state.hand_mouth_counter = max(0, state.hand_mouth_counter - 2)

    def _check_hand_to_pocket(
        self, person: PersonPose, state: BehaviorState, shoulder_width: float
    ):
        """
        Detect hand moving to hip/pocket region.
        Workers' hands stay at workbench level, not hip level, during normal work.
        """
        cfg = self.config
        left_hip = person.get_keypoint(cfg.KP_LEFT_HIP)
        right_hip = person.get_keypoint(cfg.KP_RIGHT_HIP)
        left_wrist = person.get_keypoint(cfg.KP_LEFT_WRIST)
        right_wrist = person.get_keypoint(cfg.KP_RIGHT_WRIST)

        hand_near_pocket = False

        for wrist, hip in [
            (left_wrist, left_hip),
            (right_wrist, right_hip),
        ]:
            if wrist is None or hip is None:
                continue
            dist = np.linalg.norm(np.array(wrist) - np.array(hip))
            if dist / shoulder_width < cfg.hand_pocket_ratio:
                hand_near_pocket = True
                break

        if hand_near_pocket:
            state.hand_pocket_counter += 1
            if state.hand_pocket_counter >= cfg.hand_pocket_min_frames:
                state.active_behaviors.append("HAND_TO_POCKET")
                state.anomaly_score += cfg.hand_pocket_score * 0.3
        else:
            state.hand_pocket_counter = max(0, state.hand_pocket_counter - 2)

    def _check_head_movement(
        self, person: PersonPose, state: BehaviorState, shoulder_width: float
    ):
        """
        Detect 'looking around' head scanning behavior.
        Normal factory workers are head-down focused on their work bench.
        Rapid lateral head scanning = post-theft checking behavior.
        """
        cfg = self.config
        nose = person.get_keypoint(cfg.KP_NOSE)
        left_shoulder = person.get_keypoint(cfg.KP_LEFT_SHOULDER)
        right_shoulder = person.get_keypoint(cfg.KP_RIGHT_SHOULDER)

        if nose is None or left_shoulder is None or right_shoulder is None:
            return

        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        # Normalized lateral offset of nose from shoulder center
        lateral_offset = (nose[0] - shoulder_mid_x) / shoulder_width

        state.nose_positions.append(lateral_offset)

        if len(state.nose_positions) < 10:
            return

        offsets = np.array(list(state.nose_positions))
        diffs = np.diff(offsets)
        signs = np.sign(diffs)
        signs = signs[signs != 0]   # remove flat segments

        if len(signs) < 2:
            return

        direction_changes = int(np.sum(np.diff(signs) != 0))

        if direction_changes >= cfg.head_direction_changes:
            state.active_behaviors.append("LOOKING_AROUND")
            state.anomaly_score += cfg.head_movement_score * 0.3

    # ──────────────────────────────────────────────────────────────────
    # Face Emotion Integration
    # ──────────────────────────────────────────────────────────────────

    def _integrate_face_emotion(
        self, state: BehaviorState, avg_fear: float, avg_surprise: float
    ):
        """
        Add anomaly score contributions from FER face expression results.
        Smoothed fear and surprise are already averaged over the FER window.
        """
        cfg = self.config

        if avg_fear >= cfg.fer_fear_threshold:
            state.active_behaviors.append("FEAR_DETECTED")
            state.anomaly_score += cfg.fer_fear_score * 0.3

        if avg_surprise >= cfg.fer_surprise_threshold:
            state.active_behaviors.append("SURPRISE_DETECTED")
            state.anomaly_score += cfg.fer_surprise_score * 0.3

    # ──────────────────────────────────────────────────────────────────
    # Temporal Sequence Bonus
    # ──────────────────────────────────────────────────────────────────

    def _apply_temporal_bonus(self, state: BehaviorState):
        """
        Multiply score when a concealment behavior is followed by
        a post-theft nervous behavior — the classic theft sequence.
        """
        if len(state.behavior_history) < 2:
            return

        recent_behaviors: set = set()
        for _, behaviors in list(state.behavior_history)[-10:]:
            recent_behaviors.update(behaviors)

        concealment = {"HAND_TO_MOUTH", "HAND_TO_POCKET"}
        post_theft = {"LOOKING_AROUND", "FEAR_DETECTED", "SURPRISE_DETECTED"}

        has_concealment = bool(recent_behaviors & concealment)
        has_post_theft = bool(recent_behaviors & post_theft)

        if has_concealment and has_post_theft:
            state.anomaly_score *= self.config.temporal_bonus
            if "SEQUENCE_DETECTED" not in state.active_behaviors:
                state.active_behaviors.append("SEQUENCE_DETECTED")

    # ──────────────────────────────────────────────────────────────────

    def cleanup_lost_tracks(self, active_track_ids: set):
        """Remove state for persons no longer tracked."""
        lost = [tid for tid in self.person_states if tid not in active_track_ids]
        for tid in lost:
            del self.person_states[tid]
