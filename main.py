"""
Jewelry Factory Theft Detection System — Main Pipeline (YOLOv7 + ByteTrack + FER).

Usage:
    python main.py --input input_videos/factory_cam1.mp4
    python main.py --input input_videos/ --process-fps 10 --alert-threshold 65
"""

import argparse
import cv2
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add YOLOv7 and ByteTrack repo paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "yolov7_repo"))
sys.path.insert(0, os.path.join(current_dir, "bytetrack_repo"))

from jewelry_theft_detection.config import DetectionConfig
from jewelry_theft_detection.pose_detector import PoseDetector, PersonPose
from jewelry_theft_detection.byte_tracker import ByteTrackerWrapper
from jewelry_theft_detection.behavior_analyzer import BehaviorAnalyzer
from jewelry_theft_detection.face_expression import FaceExpressionDetector
from jewelry_theft_detection.video_annotator import VideoAnnotator
from jewelry_theft_detection.alert_manager import AlertManager


def compute_iou(box_a, box_b) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_poses_to_tracks(tracked_objects, poses, iou_threshold: float = 0.3) -> dict:
    """
    Match ByteTrack bounding boxes to YOLOv7-Pose detections via IoU.

    Returns:
        Dict[track_id -> PersonPose]
    """
    matched = {}
    if not tracked_objects or not poses:
        return matched

    for obj in tracked_objects:
        track_id = obj["track_id"]
        t_bbox = obj["bbox"]
        best_iou = -1
        best_pose = None

        for pose in poses:
            iou = compute_iou(t_bbox, pose.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pose = pose

        if best_pose is not None:
            # Use stable tracked bbox instead of per-frame detection bbox
            best_pose.bbox = np.array(t_bbox)
            matched[track_id] = best_pose

    return matched


def process_video(video_path: str, output_dir: str, config: DetectionConfig):
    """Process a single video file for theft detection."""

    print(f"\n{'='*70}")
    print(f"  JEWELRY THEFT DETECTION — Processing Video (YOLOv7 + ByteTrack + FER)")
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video Info:")
    print(f"    Resolution:   {width}x{height}")
    print(f"    Source FPS:   {src_fps:.1f}")
    print(f"    Total Frames: {total_frames}")
    print(f"    Duration:     {total_frames/src_fps:.1f} seconds")
    print(f"    Process FPS:  {config.process_fps}")
    print()

    frame_skip = max(1, int(src_fps / config.process_fps))

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
    out_writer = cv2.VideoWriter(output_path, fourcc, config.output_fps, (width, height))

    # ── Initialize pipeline components ────────────────────────────
    pose_detector = PoseDetector(config)
    tracker = ByteTrackerWrapper(config)
    behavior_analyzer = BehaviorAnalyzer(config)
    face_detector = FaceExpressionDetector(config)
    annotator = VideoAnnotator(config)

    # Alert manager: saves images to alerts/ subfolder
    alert_dir = os.path.join(output_dir, "alerts")
    alert_manager = AlertManager(config, alert_dir, fps=config.process_fps)

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        processed_count += 1

        # ── Step 1: YOLOv7-Pose — detect persons + keypoints ──────
        poses = pose_detector.detect(frame)

        # ── Step 2: ByteTrack — assign/maintain track IDs ─────────
        detections = []
        for pose in poses:
            x1, y1, x2, y2 = pose.bbox
            detections.append([x1, y1, x2, y2, pose.confidence])
        detections = np.array(detections) if detections else np.empty((0, 5))

        tracked_objects = tracker.update(detections, (height, width))

        # ── Step 3: IoU matching — pose keypoints → track IDs ─────
        tracked_persons = match_poses_to_tracks(tracked_objects, poses, iou_threshold=0.3)

        # ── Step 4: Per-person analysis (behavior + face) ──────────
        behavior_states = {}
        active_ids = set(tracked_persons.keys())

        for track_id, person in tracked_persons.items():

            # FER: detect fear/surprise (smoothed over last N frames)
            avg_fear, avg_surprise = face_detector.detect(frame, track_id, person)

            # Behavior analysis with FER results fed in
            state = behavior_analyzer.analyze(
                track_id, person, processed_count,
                avg_fear=avg_fear,
                avg_surprise=avg_surprise,
            )
            behavior_states[track_id] = state

        # ── Step 5: Annotate frame ─────────────────────────────────
        annotated = annotator.annotate_frame(
            frame, tracked_persons, behavior_states,
            processed_count, config.process_fps,
        )

        # ── Step 6: Check alerts — save image on trigger ──────────
        for track_id, state in behavior_states.items():
            alert_manager.check_and_alert(
                track_id, state, processed_count, annotated
            )

        # ── Step 7: Cleanup lost tracks ────────────────────────────
        behavior_analyzer.cleanup_lost_tracks(active_ids)
        face_detector.cleanup_lost_tracks(active_ids)

        # Write annotated frame to output video
        out_writer.write(annotated)

        # Progress reporting every 20 processed frames
        if processed_count % 20 == 0:
            elapsed = time.time() - start_time
            proc_fps = processed_count / elapsed if elapsed > 0 else 0
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(
                f"  Progress: {progress:5.1f}% | "
                f"Frame: {frame_idx}/{total_frames} | "
                f"Processing: {proc_fps:.1f} FPS | "
                f"Persons: {len(tracked_persons)} | "
                f"Alerts: {alert_manager.alert_count}"
            )

    cap.release()
    out_writer.release()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  PROCESSING COMPLETE")
    print(f"  Total time:       {elapsed:.1f} seconds")
    print(f"  Frames processed: {processed_count}")
    print(f"  Avg FPS:          {processed_count/elapsed:.1f}")
    print(f"  Output video:     {output_path}")
    print(f"  Total alerts:     {alert_manager.alert_count}")
    print(f"{'='*70}\n")

    alert_manager.save_alert_log()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Jewelry Factory Theft Detection (YOLOv7 + ByteTrack + FER)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input video file or directory")
    parser.add_argument("--output", "-o", default="output_videos",
                        help="Output directory (default: output_videos)")
    parser.add_argument("--process-fps", type=int, default=10,
                        help="Processing FPS (default: 10)")
    parser.add_argument("--alert-threshold", type=float, default=65.0,
                        help="WARNING alert threshold 0-100 (default: 65)")
    parser.add_argument("--high-threshold", type=float, default=80.0,
                        help="HIGH alert threshold 0-100 (default: 80)")

    args = parser.parse_args()

    config = DetectionConfig(
        process_fps=args.process_fps,
        output_fps=args.process_fps,
        alert_threshold=args.alert_threshold,
        high_alert_threshold=args.high_threshold,
    )

    input_path = Path(args.input)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    if input_path.is_file():
        videos = [str(input_path)]
    elif input_path.is_dir():
        videos = sorted([
            str(f) for f in input_path.iterdir()
            if f.suffix.lower() in video_extensions
        ])
        if not videos:
            print(f"ERROR: No video files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(videos)} video(s) to process")
    else:
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)

    for video in videos:
        video_name = Path(video).stem
        video_output_dir = os.path.join(args.output, video_name)
        process_video(video, video_output_dir, config)

    print("\n✅ All videos processed!")


if __name__ == "__main__":
    main()
