"""
ByteTrack Tracker Adapter.
Wraps the ByteTrack implementation from bytetrack_repo to track person detections.
"""

import sys
import os
import numpy as np
from typing import List, Dict

# Add bytetrack_repo to system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "bytetrack_repo"))

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    # If the user hasn't fully setup/built bytetrack_repo, we can also fall back to importing
    # by adding path, or print a helpful error.
    print("[ByteTracker] WARNING: Could not import yolox.tracker.byte_tracker directly. Make sure setup is complete.")


class ByteTrackConfig:
    """Helper configuration object matching BYTETracker expectations."""
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class ByteTrackerWrapper:
    """Adapter for BYTETracker from ByteTrack repository."""

    def __init__(self, config):
        self.config = config
        
        # ByteTrack expects a config namespace/object with specific parameters
        bt_args = ByteTrackConfig(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh
        )
        
        self.tracker = BYTETracker(bt_args, frame_rate=config.process_fps)

    def update(self, detections: np.ndarray, info_shape: tuple) -> List[dict]:
        """
        Update the tracker with new detections.

        Args:
            detections: np.ndarray of shape (N, 5) where each row is [x1, y1, x2, y2, confidence]
            info_shape: tuple of (height, width) of the frame

        Returns:
            List of tracked objects. Each object is a dictionary:
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }
        """
        # ByteTrack expects detections as a numpy array of format [x1, y1, x2, y2, score]
        # and info_shape as (original_height, original_width)
        if len(detections) == 0:
            # Pass empty array of shape (0, 5)
            detections = np.empty((0, 5))

        # Perform tracking step
        online_targets = self.tracker.update(detections, info_shape, info_shape)

        tracked_objects = []
        for t in online_targets:
            tlbr = t.tlbr  # [top, left, bottom, right] i.e. [x1, y1, x2, y2]
            tid = t.track_id
            score = t.score
            
            tracked_objects.append({
                "track_id": int(tid),
                "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])],
                "confidence": float(score)
            })

        return tracked_objects
