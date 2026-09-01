import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class KalmanBallTracker:
    """
    Stage: Enhanced Ball Tracking using a Kalman Filter.

    Wraps around existing YOLO ball detections to:
    - Smooth noisy detections via Kalman filter
    - Interpolate the ball position when YOLO loses it (occlusion)
    - Limit unrealistic velocity jumps (ball physics constraints)

    State vector: [x, y, vx, vy] (position + velocity in pixel space)
    No new model required — operates purely on top of existing detections.
    """

    def __init__(self, max_lost_frames: int = 15, process_noise: float = 5.0, measurement_noise: float = 10.0):
        self.max_lost_frames = max_lost_frames

        # ---- Kalman Filter matrices (constant velocity model) ----
        dt = 1.0  # 1 frame time step

        # State transition matrix: x_new = F @ x_old
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Observation matrix: maps state to measurement [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Process noise covariance
        self.Q = np.eye(4) * process_noise

        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise

        # State covariance
        self.P = np.eye(4) * 100.0

        # State estimate
        self.x = None  # Will be initialized on first detection
        self.lost_count = 0
        self.is_initialized = False

    def _init_state(self, cx: float, cy: float):
        """Initialize state from first ball detection."""
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4) * 100.0
        self.is_initialized = True
        self.lost_count = 0

    def update(self, cx: Optional[float], cy: Optional[float]) -> Optional[Tuple[float, float]]:
        """
        Updates the Kalman filter with an optional ball center position.

        Args:
            cx: Ball center x in pixel space (or None if not detected)
            cy: Ball center y in pixel space (or None if not detected)

        Returns:
            Estimated (cx, cy) position, or None if tracking is lost.
        """
        if not self.is_initialized:
            if cx is None:
                return None
            self._init_state(cx, cy)
            return (cx, cy)

        # --- Predict step ---
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if cx is not None and cy is not None:
            # --- Update step (measurement available) ---
            z = np.array([[cx], [cy]], dtype=float)
            y = z - self.H @ self.x                         # Innovation
            S = self.H @ self.P @ self.H.T + self.R         # Innovation covariance
            K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
            self.lost_count = 0
        else:
            # No detection — rely on prediction only
            self.lost_count += 1

        if self.lost_count > self.max_lost_frames:
            # Too many frames without detection — reset filter
            self.is_initialized = False
            return None

        return (float(self.x[0, 0]), float(self.x[1, 0]))


def smooth_ball_trajectory(
    tracks_per_frame: List[List[Dict[str, Any]]],
    max_lost_frames: int = 15
) -> List[Optional[Dict[str, Any]]]:
    """
    Processes all tracked frames and returns a per-frame smoothed ball state.

    Returns:
        List of dicts with keys: 'cx', 'cy', 'bbox', 'interpolated' — one per frame.
        Returns None entry for frames where ball is completely lost.
    """
    kf = KalmanBallTracker(max_lost_frames=max_lost_frames)
    smoothed_ball_per_frame: List[Optional[Dict[str, Any]]] = []

    for frame_tracks in tracks_per_frame:
        # Handle single vs multiple ball candidates per frame
        ball_candidates = [t for t in frame_tracks if t['class_id'] == 3]
        ball_track = None
        
        if len(ball_candidates) == 1:
            ball_track = ball_candidates[0]
        elif len(ball_candidates) > 1:
            # Pick candidate closest to Kalman predicted state (or highest confidence)
            if kf.is_initialized and kf.x is not None:
                pred_cx, pred_cy = kf.x[0, 0], kf.x[1, 0]
                best_dist = float('inf')
                for cand in ball_candidates:
                    x1, y1, x2, y2 = cand['bbox']
                    ccx, ccy = (x1 + x2)/2.0, (y1 + y2)/2.0
                    d = (ccx - pred_cx)**2 + (ccy - pred_cy)**2
                    if d < best_dist:
                        best_dist = d
                        ball_track = cand
            else:
                ball_track = max(ball_candidates, key=lambda c: c.get('conf', 0.0))

        raw_cx, raw_cy = None, None
        raw_bbox = None
        if ball_track is not None:
            x1, y1, x2, y2 = ball_track['bbox']
            raw_cx = (x1 + x2) / 2.0
            raw_cy = (y1 + y2) / 2.0
            raw_bbox = ball_track['bbox']

        result = kf.update(raw_cx, raw_cy)

        if result is not None:
            est_cx, est_cy = result
            interpolated = (raw_cx is None)  # True if YOLO missed but KF estimated
            r = 10  # Estimated ball radius in pixels
            smoothed_ball_per_frame.append({
                'cx': est_cx,
                'cy': est_cy,
                'bbox': raw_bbox if raw_bbox else [est_cx - r, est_cy - r, est_cx + r, est_cy + r],
                'interpolated': interpolated
            })
        else:
            smoothed_ball_per_frame.append(None)

    n_interpolated = sum(1 for b in smoothed_ball_per_frame if b is not None and b['interpolated'])
    print(f"[BallTracker] Kalman smoother: {n_interpolated}/{len(tracks_per_frame)} frames interpolated (ball occluded/missing).")
    return smoothed_ball_per_frame
