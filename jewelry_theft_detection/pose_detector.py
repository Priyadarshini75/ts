"""
YOLOv7 Pose Detection Wrapper.
Handles multi-person pose estimation using YOLOv7-Pose (yolov7-w6-pose) on CPU.
"""

import sys
import os
import torch
import numpy as np

# Add yolov7_repo to path so we can import its modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov7_repo"))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from .config import DetectionConfig


class PersonPose:
    """Container for a single person's pose data."""

    def __init__(self, bbox: np.ndarray, keypoints: np.ndarray,
                 kp_conf: np.ndarray, confidence: float):
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.keypoints = keypoints    # (17, 2) - x, y for each keypoint
        self.kp_conf = kp_conf        # (17,) - confidence per keypoint
        self.confidence = confidence  # Overall detection confidence

    def get_keypoint(self, idx: int, min_conf: float = 0.3):
        """
        Get a keypoint position if its confidence is above threshold.

        Returns:
            (x, y) tuple or None if confidence is too low.
        """
        if self.kp_conf[idx] >= min_conf:
            return tuple(self.keypoints[idx])
        return None

    def get_shoulder_width(self, config: DetectionConfig) -> float:
        """
        Calculate shoulder width for scale-invariant distance thresholds.

        Returns:
            Shoulder width in pixels, or 0 if shoulders not detected.
        """
        left_shoulder = self.get_keypoint(config.KP_LEFT_SHOULDER)
        right_shoulder = self.get_keypoint(config.KP_RIGHT_SHOULDER)

        if left_shoulder is None or right_shoulder is None:
            return 0.0

        return np.sqrt(
            (left_shoulder[0] - right_shoulder[0]) ** 2 +
            (left_shoulder[1] - right_shoulder[1]) ** 2
        )

    def get_center(self) -> tuple:
        """Get center of bounding box for tracking."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PoseDetector:
    """Wraps YOLOv7-Pose for multi-person keypoint detection."""

    def __init__(self, config: DetectionConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        print(f"[PoseDetector] Loading YOLOv7 Pose model from: {config.pose_model}")
        
        # Load custom pose model
        self.model = attempt_load(config.pose_model, map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size_local(config.img_size, s=self.stride)
        print("[PoseDetector] YOLOv7 Pose model loaded successfully.")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run pose estimation on a single frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of PersonPose objects.
        """
        # Padded resize (letterbox)
        img = letterbox(frame, self.img_size, stride=self.stride, auto=True)[0]
        
        # Convert BGR to RGB, then HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.float()
        img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]

        # NMS for pose
        pred = non_max_suppression(
            pred,
            conf_thres=self.config.pose_confidence,
            iou_thres=self.config.pose_iou_threshold,
            kpt_label=True,
            nc=self.model.yaml.get('nc', 1),
            nkpt=self.model.yaml.get('nkpt', 17)
        )

        persons = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes and keypoints from img_size to original frame size
                # Bounding box is at index 0-4, keypoints start at index 6
                # Coordinates scale using scale_coords
                scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape)
                
                # Rescale keypoints
                # Keypoints format is [x, y, conf, x, y, conf, ...] starting at index 6
                # Scale x, y coords using scale_coords
                for d in det:
                    bbox = d[:4].cpu().numpy()
                    det_conf = float(d[4].cpu().numpy())
                    kpts = d[6:].cpu().numpy() # 17 * 3 = 51 values
                    
                    # Rescale keypoints coordinates back to input frame scale
                    # The keypoints are formatted as [x, y, conf, x, y, conf...]
                    keypoints = np.zeros((17, 2))
                    kp_conf = np.zeros(17)
                    
                    for k in range(17):
                        kx = kpts[k * 3]
                        ky = kpts[k * 3 + 1]
                        kconf = kpts[k * 3 + 2]
                        
                        # Rescale coordinates
                        # We project the point back based on image resize scale and padding
                        # letterbox returns: (img, ratio, (dw, dh))
                        # For simple scaling, we can use the same logic as scale_coords
                        # Let's apply standard letterbox scaling manually to keep it fast
                        # Or scale them in batch using scale_coords helper.
                        keypoints[k] = [kx, ky]
                        kp_conf[k] = kconf
                    
                    # Let's rescale the keypoints using scale_coords by forming a temp coordinate tensor
                    kpts_coords = torch.tensor(keypoints).unsqueeze(0)
                    scale_coords(img_tensor.shape[2:], kpts_coords, frame.shape)
                    keypoints = kpts_coords.squeeze(0).cpu().numpy()

                    person = PersonPose(
                        bbox=bbox,
                        keypoints=keypoints,
                        kp_conf=kp_conf,
                        confidence=det_conf,
                    )
                    persons.append(person)

        return persons


def check_img_size_local(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    import math
    return math.ceil(x / divisor) * divisor
