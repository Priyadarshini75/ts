# 💎 Jewelry Factory Theft Detection System (YOLOv7 + ByteTrack)

CPU-based suspicious behavior detection for jewelry factory workers using **YOLOv7 (person detection)**, **ByteTrack (multi-object tracking)**, and **YOLOv7-Pose** (keypoint estimation).

## 🎯 What It Detects

| Behavior | Description | How |
|----------|-------------|-----|
| **Hand to Mouth** | Worker puts hand near mouth (concealing items) | Wrist-to-nose proximity |
| **Hand to Pocket/Pants** | Worker moves hand to hip/pocket area | Wrist-to-hip proximity |
| **Dust Rubbing** | Worker rubs hands on thighs (transferring gold/diamond dust) | Repetitive wrist oscillation near thigh |
| **Looking Around** | Suspicious rapid head movement after theft | Head direction change frequency |
| **Fidgeting** | Nervous body movement | High-frequency wrist position variance |
| **Sequence Detection** | Concealment + post-theft behavior combo | Temporal pattern matching |

## 📁 Project Structure

```
td/
├── yolov7_repo/                 # Cloned YOLOv7 (pose branch)
├── bytetrack_repo/              # Cloned ByteTrack
├── weights/
│   ├── yolov7.pt                # Person detection weights
│   └── yolov7-w6-pose.pt        # Pose estimation weights
├── jewelry_theft_detection/
│   ├── __init__.py              # Package init
│   ├── config.py                # All thresholds & settings (tune here!)
│   ├── person_detector.py       # YOLOv7 person detector
│   ├── pose_detector.py         # YOLOv7-Pose keypoint estimator
│   ├── byte_tracker.py          # ByteTrack tracking adapter
│   ├── behavior_analyzer.py     # Core suspicious behavior detection
│   ├── video_annotator.py       # Draw overlays on output video
│   └── alert_manager.py         # Alert logging & clip saving
├── input_videos/                # Place your factory videos here
├── output_videos/               # Annotated output videos saved here
├── main.py                      # Entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install loguru cython-bbox lap filterpy
```

### 2. Run Detection

```bash
# Process a single video
python main.py --input input_videos/factory_cam1.mp4

# Process all videos in a directory
python main.py --input input_videos/

# Lower FPS for slower CPUs
python main.py --input video.mp4 --process-fps 5

# Adjust sensitivity (lower = more alerts)
python main.py --input video.mp4 --alert-threshold 40
```

## ⚙️ Configuration

All thresholds are in `jewelry_theft_detection/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `process_fps` | 10 | Frames processed per second (lower = faster) |
| `alert_threshold` | 50.0 | Score above this triggers WARNING alert |
| `high_alert_threshold` | 75.0 | Score above this triggers HIGH alert |
| `hand_mouth_ratio` | 0.6 | Hand-to-mouth distance threshold (fraction of shoulder width) |
| `hand_pocket_ratio` | 0.5 | Hand-to-pocket distance threshold |
| `head_direction_changes` | 3 | Min head direction reversals to flag "looking around" |
