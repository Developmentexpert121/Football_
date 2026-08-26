"""
Stage F: Action Recognition — Rule-based Player Action Classification.

Classifies what each player is doing per frame using pose keypoints from
Stage C (PoseEstimator) and speed data from Stage E (AnalyticsEngine):

    | Action          | Detection Method                                  |
    |-----------------|---------------------------------------------------|
    | Standing        | Speed < 1.5 km/h                                  |
    | Walking         | Speed 1.5 - 7.0 km/h                              |
    | Running         | Speed 7.0 - 25.0 km/h + leg angle oscillation      |
    | Sprinting       | Speed > 25.0 km/h                                  |
    | Kicking/Shooting| Ankle velocity spike + leg extension angle          |
    | Sliding Tackle  | Hip-to-ground proximity (hip Y near ankle Y)        |
    | Header          | Head keypoint + ball proximity at head height       |
    | Dribbling       | Running + ball possession within 2m                 |

No external model required — uses rule-based logic over pose keypoints and speed.
Future: VideoMAE deep learning model for fine-grained action recognition.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# Action labels
ACTION_STANDING = 'Standing'
ACTION_WALKING = 'Walking'
ACTION_RUNNING = 'Running'
ACTION_SPRINTING = 'Sprinting'
ACTION_KICKING = 'Kicking'
ACTION_SLIDING_TACKLE = 'Sliding Tackle'
ACTION_HEADER = 'Header'
ACTION_DRIBBLING = 'Dribbling'
ACTION_UNKNOWN = 'Unknown'

# Speed thresholds (km/h)
SPEED_STANDING_MAX = 1.5
SPEED_WALKING_MAX = 7.0
SPEED_RUNNING_MAX = 25.0
# Above 25.0 km/h = Sprinting

# Keypoint indices (COCO 17)
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16


class ActionRecognizer:
    """
    Classifies player actions per frame using pose keypoints + speed.

    Pipeline integration:
        1. Receive pose_per_frame from PoseEstimator
        2. Receive speed data from AnalyticsEngine
        3. Optionally receive ball position for proximity-based actions (header, dribbling)
    """

    def __init__(
        self,
        kick_ankle_velocity_thresh: float = 40.0,
        kick_leg_angle_thresh: float = 140.0,
        tackle_hip_ground_ratio: float = 0.75,
        header_ball_head_dist_px: float = 60.0,
        dribble_ball_dist_m: float = 2.0
    ):
        """
        Args:
            kick_ankle_velocity_thresh: Min ankle velocity (px/frame) to flag kick
            kick_leg_angle_thresh: Min knee extension angle (degrees) for kick
            tackle_hip_ground_ratio: When hip Y is > this ratio of bbox height → tackle
            header_ball_head_dist_px: Max pixel distance between head and ball for header
            dribble_ball_dist_m: Max metric distance to ball while running → dribbling
        """
        self.kick_ankle_vel_thresh = kick_ankle_velocity_thresh
        self.kick_leg_angle_thresh = kick_leg_angle_thresh
        self.tackle_hip_ground_ratio = tackle_hip_ground_ratio
        self.header_ball_head_dist = header_ball_head_dist_px
        self.dribble_ball_dist = dribble_ball_dist_m

        # Previous frame keypoints for velocity computation
        self._prev_keypoints: Dict[int, np.ndarray] = {}

    def classify_actions(
        self,
        pose_per_frame: List[Dict[int, Dict[str, Any]]],
        player_speeds_per_frame: List[Dict[int, float]],
        ball_positions_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None,
        metric_positions_per_frame: Optional[List[Dict[int, Tuple[float, float]]]] = None,
        ball_metric_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None
    ) -> List[Dict[int, str]]:
        """
        Classifies the action of each tracked player for every frame.

        Args:
            pose_per_frame: From PoseEstimator — track_id → keypoints dict
            player_speeds_per_frame: track_id → instantaneous speed (km/h)
            ball_positions_per_frame: Ball (cx, cy) in pixel space per frame (optional)
            metric_positions_per_frame: track_id → (x_m, y_m) per frame (optional, for dribble)
            ball_metric_per_frame: Ball (x_m, y_m) per frame (optional, for dribble)

        Returns:
            List of dicts, one per frame: track_id → action_label (str)
        """
        actions_per_frame: List[Dict[int, str]] = []
        self._prev_keypoints = {}

        for frame_idx in range(len(pose_per_frame)):
            frame_actions: Dict[int, str] = {}
            pose_data = pose_per_frame[frame_idx]
            speeds = player_speeds_per_frame[frame_idx] if frame_idx < len(player_speeds_per_frame) else {}

            ball_px = None
            if ball_positions_per_frame and frame_idx < len(ball_positions_per_frame):
                ball_px = ball_positions_per_frame[frame_idx]

            ball_m = None
            if ball_metric_per_frame and frame_idx < len(ball_metric_per_frame):
                ball_m = ball_metric_per_frame[frame_idx]

            metric_pos = {}
            if metric_positions_per_frame and frame_idx < len(metric_positions_per_frame):
                metric_pos = metric_positions_per_frame[frame_idx]

            for track_id, pdata in pose_data.items():
                kps = pdata.get('keypoints')
                if kps is None:
                    frame_actions[track_id] = ACTION_UNKNOWN
                    continue

                speed_kmh = speeds.get(track_id, 0.0)
                action = self._classify_single_player(
                    track_id, kps, speed_kmh, ball_px, ball_m,
                    metric_pos.get(track_id)
                )
                frame_actions[track_id] = action

                # Store for next-frame velocity
                self._prev_keypoints[track_id] = kps.copy()

            actions_per_frame.append(frame_actions)

        # Summary stats
        action_counts: Dict[str, int] = {}
        for frame_acts in actions_per_frame:
            for act in frame_acts.values():
                action_counts[act] = action_counts.get(act, 0) + 1
        print(f"[ActionRecognizer] Action distribution: {action_counts}")

        return actions_per_frame

    def _classify_single_player(
        self,
        track_id: int,
        kps: np.ndarray,
        speed_kmh: float,
        ball_px: Optional[Tuple[float, float]],
        ball_m: Optional[Tuple[float, float]],
        player_m: Optional[Tuple[float, float]]
    ) -> str:
        """
        Classifies action for a single player using pose + speed heuristics.

        Priority order (highest confidence first):
        1. Sliding Tackle (body low to ground)
        2. Header (head near ball at head height)
        3. Kicking/Shooting (ankle velocity + leg extension)
        4. Dribbling (running + near ball)
        5. Sprinting / Running / Walking / Standing (speed-based)
        """

        # --- Check: Sliding Tackle ---
        if self._is_sliding_tackle(kps):
            return ACTION_SLIDING_TACKLE

        # --- Check: Header ---
        if ball_px is not None and self._is_header(kps, ball_px):
            return ACTION_HEADER

        # --- Check: Kicking / Shooting ---
        if self._is_kicking(track_id, kps):
            return ACTION_KICKING

        # --- Check: Dribbling (running + ball possession) ---
        if speed_kmh > SPEED_WALKING_MAX and ball_m is not None and player_m is not None:
            dist_to_ball = np.sqrt(
                (ball_m[0] - player_m[0]) ** 2 + (ball_m[1] - player_m[1]) ** 2
            )
            if dist_to_ball < self.dribble_ball_dist:
                return ACTION_DRIBBLING

        # --- Speed-based classification (fallback) ---
        if speed_kmh > SPEED_RUNNING_MAX:
            return ACTION_SPRINTING
        elif speed_kmh > SPEED_WALKING_MAX:
            return ACTION_RUNNING
        elif speed_kmh > SPEED_STANDING_MAX:
            return ACTION_WALKING
        else:
            return ACTION_STANDING

    def _is_sliding_tackle(self, kps: np.ndarray) -> bool:
        """
        Detects sliding tackle: hips are close to ankle height (body nearly horizontal).

        Logic: If the vertical distance between mid-hip and mid-ankle is less than
        tackle_hip_ground_ratio × the vertical span of the whole body → tackle.
        """
        left_hip = kps[KP_LEFT_HIP]
        right_hip = kps[KP_RIGHT_HIP]
        left_ankle = kps[KP_LEFT_ANKLE]
        right_ankle = kps[KP_RIGHT_ANKLE]
        nose = kps[KP_NOSE]

        # Need sufficient keypoint confidence
        if min(left_hip[2], right_hip[2], left_ankle[2], right_ankle[2], nose[2]) < 0.3:
            return False

        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        mid_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        body_span_y = abs(mid_ankle_y - nose[1])

        if body_span_y < 20:  # Too small to determine
            return False

        hip_to_ankle_ratio = abs(mid_ankle_y - mid_hip_y) / body_span_y

        # If hips are very close to ankle height (body is near-horizontal)
        return hip_to_ankle_ratio < (1.0 - self.tackle_hip_ground_ratio)

    def _is_header(self, kps: np.ndarray, ball_px: Tuple[float, float]) -> bool:
        """
        Detects header: nose/head keypoint is near the ball and above shoulder height.
        """
        nose = kps[KP_NOSE]
        left_shoulder = kps[KP_LEFT_SHOULDER]
        right_shoulder = kps[KP_RIGHT_SHOULDER]

        if nose[2] < 0.3:
            return False

        # Ball proximity to head
        head_ball_dist = np.sqrt(
            (nose[0] - ball_px[0]) ** 2 + (nose[1] - ball_px[1]) ** 2
        )

        if head_ball_dist > self.header_ball_head_dist:
            return False

        # Ball should be at or above shoulder height (lower Y = higher position in image)
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2 if min(left_shoulder[2], right_shoulder[2]) > 0.3 else nose[1] + 30
        if ball_px[1] > avg_shoulder_y + 20:
            return False

        return True

    def _is_kicking(self, track_id: int, kps: np.ndarray) -> bool:
        """
        Detects kicking/shooting: ankle has high velocity + knee is extended (large angle).

        Computes ankle velocity by comparing current keypoints to previous frame.
        """
        prev_kps = self._prev_keypoints.get(track_id)
        if prev_kps is None:
            return False

        # Check both legs
        for ankle_idx, knee_idx, hip_idx in [
            (KP_LEFT_ANKLE, KP_LEFT_KNEE, KP_LEFT_HIP),
            (KP_RIGHT_ANKLE, KP_RIGHT_KNEE, KP_RIGHT_HIP)
        ]:
            ankle_cur = kps[ankle_idx]
            ankle_prev = prev_kps[ankle_idx]
            knee = kps[knee_idx]
            hip = kps[hip_idx]

            if min(ankle_cur[2], ankle_prev[2], knee[2], hip[2]) < 0.3:
                continue

            # Ankle velocity (pixels per frame)
            ankle_vel = np.sqrt(
                (ankle_cur[0] - ankle_prev[0]) ** 2 +
                (ankle_cur[1] - ankle_prev[1]) ** 2
            )

            if ankle_vel < self.kick_ankle_vel_thresh:
                continue

            # Knee extension angle (hip → knee → ankle)
            angle = self._joint_angle(
                kps[hip_idx, :2], kps[knee_idx, :2], kps[ankle_idx, :2]
            )

            if angle > self.kick_leg_angle_thresh:
                return True

        return False

    @staticmethod
    def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Computes angle at joint b (in degrees) formed by segments a→b and b→c."""
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
