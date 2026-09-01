"""
Stage C: Pose Estimation using YOLOv8 Pose.

Extracts 17 COCO body keypoints per detected player per frame:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Uses yolov8n-pose.pt (smallest, fastest variant — auto-downloaded by ultralytics).
No training required. Works on CPU and GPU.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Graceful fallback if ultralytics is not installed or model load fails
try:
    from ultralytics import YOLO
    POSE_MODEL_AVAILABLE = True
except ImportError:
    POSE_MODEL_AVAILABLE = False

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton edge connections for drawing
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Face
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                           # Left arm
    (6, 8), (8, 10),                          # Right arm
    (5, 11), (6, 12),                         # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                       # Left leg
    (12, 14), (14, 16),                       # Right leg
]


class PoseEstimator:
    """
    Extracts body pose keypoints from video frames using YOLOv8-Pose.

    For each frame, runs pose inference and matches detected poses to existing
    tracker bounding boxes via IoU overlap. Each player track gets a set of
    17 keypoints with (x, y, confidence) per joint.
    """

    def __init__(self, model_path: str = 'yolov8n-pose.pt', conf_thresh: float = 0.3, device: str = 'auto'):
        self.model = None
        self._available = POSE_MODEL_AVAILABLE
        self.conf_thresh = conf_thresh

        if not self._available:
            print("[PoseEstimator] ultralytics not installed. Pose estimation disabled.")
            return

        try:
            if device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model = YOLO(model_path)
            self.device = device
            print(f"[PoseEstimator] YOLOv8 Pose model loaded: {model_path} (device={device})")
        except Exception as e:
            print(f"[PoseEstimator] Warning: Failed to load pose model ({e}). Pose estimation disabled.")
            self._available = False

    def estimate_poses(
        self,
        frames: List[np.ndarray],
        tracks_per_frame: List[List[Dict[str, Any]]],
        process_every_n: int = 1
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Runs pose estimation on frames and matches keypoints to tracked players.

        Args:
            frames: List of video frames (BGR numpy arrays)
            tracks_per_frame: Tracker output per frame
            process_every_n: Run pose inference every N frames (1 = every frame)

        Returns:
            List of dicts, one per frame. Each dict maps track_id -> {
                'keypoints': np.ndarray of shape (17, 3) — [x, y, conf],
                'keypoint_names': list of str,
                'bbox': [x1, y1, x2, y2]  (pose model bbox)
            }
        """
        if not self._available or self.model is None:
            return [{} for _ in frames]

        pose_per_frame: List[Dict[int, Dict[str, Any]]] = []
        last_pose_data: Dict[int, Dict[str, Any]] = {}

        for frame_idx, frame in enumerate(frames):
            if frame_idx % process_every_n != 0:
                # Reuse last computed pose data for skipped frames
                pose_per_frame.append(last_pose_data.copy())
                continue

            tracks = tracks_per_frame[frame_idx]
            # Goalkeeper-Only Pose Filter: filter for class_id == 1
            gk_tracks = [t for t in tracks if t['class_id'] == 1]
            if not gk_tracks:
                pose_per_frame.append({})
                continue

            try:
                results = self.model.predict(
                    frame,
                    conf=self.conf_thresh,
                    device=self.device,
                    verbose=False
                )
            except Exception:
                pose_per_frame.append({})
                continue

            frame_poses: Dict[int, Dict[str, Any]] = {}

            if results and len(results) > 0:
                result = results[0]

                # Extract pose bboxes and keypoints from YOLO result
                if result.keypoints is not None and result.boxes is not None:
                    pose_bboxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                    keypoints_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

                    # Match tracker Goalkeeper bboxes to pose bboxes
                    for track in gk_tracks:

                        t_id = track['track_id']
                        t_bbox = track['bbox']

                        best_iou = 0.3  # Minimum threshold for match
                        best_idx = -1

                        for p_idx in range(len(pose_bboxes)):
                            iou = self._compute_iou(t_bbox, pose_bboxes[p_idx].tolist())
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = p_idx

                        if best_idx >= 0:
                            kps = keypoints_data[best_idx]  # (17, 3)
                            frame_poses[t_id] = {
                                'keypoints': kps,
                                'keypoint_names': KEYPOINT_NAMES,
                                'bbox': pose_bboxes[best_idx].tolist(),
                                'match_iou': best_iou
                            }

            pose_per_frame.append(frame_poses)
            last_pose_data = frame_poses

        n_frames_with_poses = sum(1 for p in pose_per_frame if p)
        print(f"[PoseEstimator] Extracted poses for {n_frames_with_poses}/{len(frames)} frames.")
        return pose_per_frame

    @staticmethod
    def get_keypoint(keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float, float]]:
        """
        Utility to get a specific keypoint by name.

        Returns:
            (x, y, confidence) or None if keypoint confidence is too low.
        """
        if name not in KEYPOINT_NAMES:
            return None
        idx = KEYPOINT_NAMES.index(name)
        x, y, conf = keypoints[idx]
        if conf < 0.3:
            return None
        return (float(x), float(y), float(conf))

    @staticmethod
    def get_joint_angle(kps: np.ndarray, joint_a_idx: int, joint_b_idx: int, joint_c_idx: int) -> float:
        """
        Computes the angle at joint_b formed by the limb segments A→B and B→C.
        Used for action recognition (e.g., knee angle for kicking detection).

        Returns:
            Angle in degrees (0-180), or -1.0 if any keypoint has low confidence.
        """
        if kps[joint_a_idx, 2] < 0.3 or kps[joint_b_idx, 2] < 0.3 or kps[joint_c_idx, 2] < 0.3:
            return -1.0

        a = kps[joint_a_idx, :2]
        b = kps[joint_b_idx, :2]
        c = kps[joint_c_idx, :2]

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        return float(angle)

    @staticmethod
    def _compute_iou(boxA: list, boxB: list) -> float:
        """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0
