"""
Stage 9 (Enhanced): Comprehensive Event Detection Rule Engine.

Detects the following match events from spatial/temporal analysis:

    | Event              | Detection Method                                       |
    |--------------------|-------------------------------------------------------|
    | Potential Foul     | Opposing player collision proximity < threshold        |
    | Yellow Card        | Referee close to player after foul flagged             |
    | Goal               | Ball crosses goal line at goal mouth + celebration     |
    | Corner Kick        | Ball exits near corner flag area                       |
    | Free Kick          | Ball stationary after foul + player cluster            |
    | Offside            | Attacker ahead of 2nd-last defender at pass moment     |
    | Shot on Target     | Ball velocity spike toward goal direction               |
    | Penalty Area Entry | Ball enters 18-yard box with attacking player          |

All detection is rule-based from ball trajectory + player positions + field geometry.
No additional model required.
"""

import cv2
import numpy as np
import os
from collections import deque
from typing import List, Dict, Any, Tuple, Optional


class BallTrajectoryBuffer:
    """
    Maintains a rolling window of ball states (pixel pos + ground pos +
    bbox size) so we can:
      1. Save ±N frames around a confirmed goal (clip evidence).
      2. Extrapolate trajectory into goal when ball is occluded.
      3. Estimate airborne height from apparent diameter shrinkage.
    """

    WINDOW = 15          # keep last 15 frames (±5 @ 25fps with headroom)
    CROSSBAR_H_M = 2.44  # FIFA crossbar height in metres
    BALL_DIAM_M  = 0.22  # FIFA ball diameter in metres

    def __init__(self, ground_diameter_px: float = 28.0):
        """
        ground_diameter_px: calibrated pixel diameter of the ball
        when it is rolling on the pitch centre.
        """
        self._buf: deque = deque(maxlen=self.WINDOW)
        self.ground_diam_px = ground_diameter_px  # D_ground

    def push(self, frame_idx: int,
             ball_px: Optional[Tuple[float, float]],
             ground_pos: Optional[Tuple[float, float]],
             bbox: Optional[Tuple[float, float, float, float]]):
        """
        Call once per frame inside detect_events BEFORE the goal check.
        bbox = (x1, y1, x2, y2) in pixel space.
        """
        diam_px = None
        if bbox is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            diam_px = (w + h) / 2.0   # average of width & height

        self._buf.append({
            'frame_idx': frame_idx,
            'ball_px':   ball_px,
            'ground':    ground_pos,
            'diam_px':   diam_px,
        })

    def estimate_height_m(self) -> float:
        """
        Estimate ball height above ground (Z) from apparent diameter
        shrinkage vs the ground-level calibration diameter.
        """
        if not self._buf:
            return 0.0
        latest = self._buf[-1]
        if latest['ball_px'] is None:
            return 0.0
        d = latest['diam_px']
        if d is None or d <= 0 or self.ground_diam_px <= 0:
            return 0.0

        ratio = self.ground_diam_px / d
        K = 1.5   # tune for camera rig
        z_est = max(0.0, (ratio - 1.0) * K)
        return min(z_est, self.CROSSBAR_H_M + 0.30)   # 30 cm tolerance

    def is_airborne(self) -> bool:
        return self.estimate_height_m() > 0.25   # >25 cm off ground

    def extrapolate_will_cross(self, goal_line_x: float,
                                goal_y_min: float, goal_y_max: float,
                                look_ahead: int = 7) -> Tuple[bool, float]:
        """
        Fit a linear trajectory through the last 5 ground positions and
        check whether it crosses the goal line within `look_ahead` frames.

        Returns (will_cross: bool, confidence: float 0-1).
        """
        pts = [(e['ground'][0], e['ground'][1])
               for e in self._buf
               if e['ground'] is not None]

        if len(pts) < 3:
            return False, 0.0

        pts = pts[-5:]   # use last 5 valid positions
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        t  = np.arange(len(xs), dtype=float)

        try:
            vx = float(np.polyfit(t, xs, 1)[0])   # dx per frame
            vy = float(np.polyfit(t, ys, 1)[0])   # dy per frame
        except Exception:
            return False, 0.0

        if abs(vx) < 1e-4:
            return False, 0.0

        frames_to_line = (goal_line_x - xs[-1]) / vx
        if not (0 < frames_to_line <= look_ahead):
            return False, 0.0

        y_at_line = ys[-1] + vy * frames_to_line
        if not (goal_y_min <= y_at_line <= goal_y_max):
            return False, 0.0

        confidence = 1.0 - (frames_to_line / look_ahead)
        return True, round(confidence, 2)

    def get_window_frames(self, goal_frame_idx: int,
                          half_window: int = 5) -> List[Dict]:
        """
        Return buffered entries within ±half_window of goal_frame_idx.
        """
        lo = goal_frame_idx - half_window
        hi = goal_frame_idx + half_window
        return [e for e in self._buf if lo <= e['frame_idx'] <= hi]


class NetRippleDetector:
    """
    Detects net motion (optical flow magnitude spike) inside a
    configurable ROI around each goalmouth.
    """

    SPIKE_RATIO   = 3.0   # motion must be 3× baseline to count
    SPIKE_ABS_MIN = 1.0   # absolute flow magnitude floor
    BUFFER_LEN    = 12    # rolling history window (frames)

    def __init__(self, left_roi: Tuple[int,int,int,int],
                       right_roi: Tuple[int,int,int,int]):
        self.rois = {'left': left_roi, 'right': right_roi}
        self._prev_gray: Optional[np.ndarray] = None
        self._history: Dict[str, deque] = {
            'left':  deque(maxlen=self.BUFFER_LEN),
            'right': deque(maxlen=self.BUFFER_LEN),
        }

    def update(self, frame_bgr: np.ndarray,
               side: str) -> float:
        """
        Call every frame. Returns ripple confidence ∈ [0, 1].
        `side` = 'left' | 'right'
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray.copy()
            return 0.0

        x1, y1, x2, y2 = self.rois[side]
        h, w = gray.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))

        curr_roi = gray[y1:y2, x1:x2]
        prev_roi = self._prev_gray[y1:y2, x1:x2]

        if curr_roi.size == 0 or prev_roi.size == 0:
            self._prev_gray = gray.copy()
            return 0.0

        # Farneback dense optical flow on the net ROI
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, curr_roi,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        score = float(np.mean(mag))

        self._prev_gray = gray.copy()
        hist = self._history[side]
        hist.append(score)

        if len(hist) < 4:
            return 0.0

        recent  = max(list(hist)[-3:])
        baseline = float(np.mean(list(hist)[:-3])) if len(hist) > 3 else 0.0

        spike = (recent > baseline * self.SPIKE_RATIO and
                 recent > self.SPIKE_ABS_MIN)

        if spike:
            return min(1.0, recent / 5.0)   # normalised confidence
        return 0.0


class EventDetector:
    """
    Enhanced Event Detection Engine — flags match events from tracked positions.
    """
    def __init__(
        self,
        fps: float = 25.0,
        foul_dist_thresh: float = 1.2,
        card_referee_dist_thresh: float = 1.5,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        goal_line_thresh: float = 1.8,
        goal_mouth_y_min: float = 28.0,
        goal_mouth_y_max: float = 40.0,
        penalty_area_depth: float = 16.5,
        penalty_area_width_half: float = 20.16,
        corner_area_radius: float = 5.0,
        shot_velocity_thresh: float = 8.0,
        offside_margin: float = 0.5,
        left_goal_polygon: Optional[List[List[float]]] = None,
        right_goal_polygon: Optional[List[List[float]]] = None,
        consecutive_goal_frames: int = 2,
        ball_ground_diameter_px: float = 28.0,
        left_net_roi: Tuple[int, int, int, int] = (150, 220, 490, 590),
        right_net_roi: Tuple[int, int, int, int] = (690, 240, 1040, 610)
    ):
        self.fps = fps
        self.foul_dist_thresh = foul_dist_thresh
        self.card_referee_dist_thresh = card_referee_dist_thresh
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Goal detection thresholds
        self.goal_line_thresh = goal_line_thresh  # meters from goal line
        self.goal_mouth_y_min = goal_mouth_y_min  # 7.32m goal width centered on 68m pitch
        self.goal_mouth_y_max = goal_mouth_y_max

        # 2.5D Hybrid Goalmouth Polygons (Image Space P1, P2, P3, P4)
        self.left_goal_polygon = left_goal_polygon or [
            [180, 580], [480, 420], [480, 260], [180, 250]
        ]
        self.right_goal_polygon = right_goal_polygon or [
            [700, 440], [1020, 600], [1020, 260], [700, 270]
        ]
        self.consecutive_goal_frames = consecutive_goal_frames
        self._goal_streak = 0

        # Goal Detection Upgrade: new components
        self.ball_buffer = BallTrajectoryBuffer(ground_diameter_px=ball_ground_diameter_px)
        self.net_ripple = NetRippleDetector(
            left_roi=left_net_roi, right_roi=right_net_roi
        )

        # Penalty area: 16.5m deep, 40.32m wide (centered)
        self.penalty_area_depth = penalty_area_depth
        self.penalty_area_y_min = (pitch_width - 2 * penalty_area_width_half) / 2
        self.penalty_area_y_max = pitch_width - self.penalty_area_y_min

        # Corner and shot thresholds
        self.corner_area_radius = corner_area_radius
        self.shot_velocity_thresh = shot_velocity_thresh  # m/s ball velocity toward goal
        self.offside_margin = offside_margin  # meters

        # Ball velocity history for shot detection
        self._ball_velocities: List[Tuple[float, float]] = []

    def detect_events(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int],
        ball_metric_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None,
        jersey_map: Optional[Dict[int, str]] = None,
        ball_pixels_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None,
        raw_frames: Optional[List[np.ndarray]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates sequence of frames for all event types.

        Returns:
            List of event dictionaries sorted by timestamp.
        """
        events = []
        num_frames = len(metric_positions_per_frame)

        # Precompute ball velocity if ball positions available
        ball_velocities = self._compute_ball_velocities(ball_metric_per_frame)

        for frame_idx in range(1, num_frames - 1):
            pos_dict = metric_positions_per_frame[frame_idx]
            tracks = tracks_per_frame[frame_idx]
            timestamp_sec = round(frame_idx / self.fps, 2)
            timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"

            # Check for OCR-Assisted Zoom/Slow-Mo Replay
            if jersey_map and self._is_zoom_replay(tracks, tracks_per_frame[frame_idx - 1], jersey_map, events):
                continue

            # Separate players by team and locate referee/ball
            players_team_0 = []
            players_team_1 = []
            referee_pos = None
            ball_pos = None

            for track in tracks:
                t_id = track['track_id']
                cls_id = track['class_id']
                if t_id not in pos_dict:
                    continue

                pos = pos_dict[t_id]
                if cls_id == 0:
                    team = team_assignments.get(t_id, 0)
                    if team == 0:
                        players_team_0.append((t_id, pos))
                    else:
                        players_team_1.append((t_id, pos))
                elif cls_id == 2:  # Referee
                    referee_pos = pos
                elif cls_id == 3:  # Ball
                    ball_pos = pos

            # Override ball_pos with smoothed metric position if available
            if ball_metric_per_frame and frame_idx < len(ball_metric_per_frame):
                if ball_metric_per_frame[frame_idx] is not None:
                    ball_pos = ball_metric_per_frame[frame_idx]

            # ===== 1. Potential Foul Detection =====
            for p0_id, p0_pos in players_team_0:
                for p1_id, p1_pos in players_team_1:
                    dist = np.sqrt((p0_pos[0] - p1_pos[0])**2 + (p0_pos[1] - p1_pos[1])**2)
                    if dist <= self.foul_dist_thresh:
                        events.append({
                            'frame_idx': frame_idx,
                            'timestamp': timestamp_str,
                            'timestamp_seconds': timestamp_sec,
                            'event_type': 'Potential Foul',
                            'players_involved': [p0_id, p1_id],
                            'teams_involved': [0, 1],
                            'confidence': round(min(0.95, 1.0 - (dist / self.foul_dist_thresh) * 0.4), 2),
                            'description': f"Player #{p0_id} (Team A) and Player #{p1_id} (Team B) tight collision ({round(dist, 2)}m)"
                        })

            # ===== 2. Yellow Card Candidate =====
            if referee_pos is not None:
                for p_id, p_pos in players_team_0 + players_team_1:
                    dist_to_ref = np.sqrt((referee_pos[0] - p_pos[0])**2 + (referee_pos[1] - p_pos[1])**2)
                    if dist_to_ref <= self.card_referee_dist_thresh:
                        near_foul = any(
                            abs(e['frame_idx'] - frame_idx) < int(self.fps * 2) and p_id in e['players_involved']
                            for e in events if e['event_type'] == 'Potential Foul'
                        )
                        if near_foul:
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Yellow Card Candidate',
                                'players_involved': [p_id],
                                'teams_involved': [team_assignments.get(p_id, 0)],
                                'confidence': 0.88,
                                'description': f"Referee intervention with Player #{p_id} following collision."
                            })

            # ===== 3. Goal Detection (5-Signal Fusion Engine) =====

            # Push ball state into rolling buffer FIRST
            curr_ball_px   = (ball_pixels_per_frame[frame_idx]
                              if (ball_pixels_per_frame and
                                  frame_idx < len(ball_pixels_per_frame))
                              else None)
            curr_ball_bbox = None
            for tr in tracks:
                if tr['class_id'] == 3:           # ball class
                    curr_ball_bbox = tuple(tr['bbox'])
                    break

            self.ball_buffer.push(
                frame_idx, curr_ball_px, ball_pos, curr_ball_bbox
            )

            if ball_pos is not None:

                ball_vel = (ball_velocities[frame_idx]
                            if (frame_idx < len(ball_velocities) and
                                ball_velocities[frame_idx] is not None)
                            else (0.0, 0.0))

                # ── Net ripple (needs raw frame) ─────────────
                net_ripple_left  = 0.0
                net_ripple_right = 0.0
                if (raw_frames is not None and
                        frame_idx < len(raw_frames) and
                        raw_frames[frame_idx] is not None):
                    frame_bgr = raw_frames[frame_idx]
                    net_ripple_left  = self.net_ripple.update(frame_bgr, 'left')
                    net_ripple_right = self.net_ripple.update(frame_bgr, 'right')

                # ── Trajectory extrapolation ─────────────────
                traj_right, traj_right_conf = self.ball_buffer.extrapolate_will_cross(
                    goal_line_x = self.pitch_length - self.goal_line_thresh,
                    goal_y_min  = self.goal_mouth_y_min,
                    goal_y_max  = self.goal_mouth_y_max,
                )
                traj_left, traj_left_conf = self.ball_buffer.extrapolate_will_cross(
                    goal_line_x = self.goal_line_thresh,
                    goal_y_min  = self.goal_mouth_y_min,
                    goal_y_max  = self.goal_mouth_y_max,
                )

                # ── Left goal check ──────────────────────────
                is_goal_left, conf_left, desc_left = self._check_goal_hybrid(
                    ball_pixel      = curr_ball_px,
                    ground_pos      = ball_pos,
                    goal_side       = 'left',
                    goal_polygon    = self.left_goal_polygon,
                    ball_velocity   = ball_vel,
                    ball_bbox       = curr_ball_bbox,
                    net_ripple_conf = net_ripple_left,
                    traj_cross_conf = traj_left_conf if traj_left else 0.0,
                )

                # ── Right goal check ─────────────────────────
                is_goal_right, conf_right, desc_right = self._check_goal_hybrid(
                    ball_pixel      = curr_ball_px,
                    ground_pos      = ball_pos,
                    goal_side       = 'right',
                    goal_polygon    = self.right_goal_polygon,
                    ball_velocity   = ball_vel,
                    ball_bbox       = curr_ball_bbox,
                    net_ripple_conf = net_ripple_right,
                    traj_cross_conf = traj_right_conf if traj_right else 0.0,
                )

                if is_goal_left or is_goal_right:
                    self._goal_streak += 1
                    conf     = conf_left   if is_goal_left  else conf_right
                    desc     = desc_left   if is_goal_left  else desc_right
                    goal_side_str = 'left' if is_goal_left  else 'right'

                    if self._goal_streak >= self.consecutive_goal_frames:
                        # ── Shooter attribution ───────────────
                        all_players   = players_team_0 + players_team_1
                        scorer_id     = None
                        min_dist_ball = 999.0
                        for p_id, p_pos in all_players:
                            d = np.sqrt((p_pos[0]-ball_pos[0])**2 +
                                        (p_pos[1]-ball_pos[1])**2)
                            if d < min_dist_ball:
                                min_dist_ball = d
                                scorer_id     = p_id

                        if scorer_id is not None:
                            scoring_team = team_assignments.get(scorer_id,
                                           1 if is_goal_left else 0)
                        else:
                            scoring_team = 1 if is_goal_left else 0

                        conceding_team = 1 - scoring_team
                        scorer_j = (jersey_map.get(scorer_id, scorer_id)
                                    if (jersey_map and scorer_id is not None and scorer_id in jersey_map) else (scorer_id if scorer_id is not None else "Unknown"))

                        # ── Cooldown guard ────────────────────
                        recent_goal = any(
                            abs(e['frame_idx'] - frame_idx) < int(self.fps * 20.0)
                            and e['event_type'] == 'Goal'
                            for e in events
                        )
                        if not recent_goal:

                            # ── ±5 frame evidence window ──────
                            evidence_frames = self.ball_buffer.get_window_frames(
                                frame_idx, half_window=5
                            )
                            evidence_summary = [
                                {
                                    'frame': e['frame_idx'],
                                    'ground': (round(e['ground'][0], 2),
                                               round(e['ground'][1], 2))
                                    if e['ground'] else None,
                                    'px': e['ball_px'],
                                }
                                for e in evidence_frames
                            ]

                            scorer_desc = (
                                f" | Goal Scored by Team {scoring_team+1} "
                                f"(Player #{scorer_j}) "
                                f"| Conceded by Team {conceding_team+1}"
                            )

                            events.append({
                                'frame_idx':          frame_idx,
                                'timestamp':          timestamp_str,
                                'timestamp_seconds':  timestamp_sec,
                                'event_type':         'Goal',
                                'players_involved':   [scorer_id] if scorer_id is not None else [],
                                'teams_involved':     [scoring_team],
                                'conceding_team':     conceding_team,
                                'confidence':         conf,
                                'goal_side':          goal_side_str,
                                'description':        desc + scorer_desc,
                                'evidence_frames':    evidence_summary,
                                'net_ripple_left':    net_ripple_left,
                                'net_ripple_right':   net_ripple_right,
                            })

                            print(
                                f"\n⚽ [GOAL CONFIRMED] {timestamp_str} "
                                f"(Frame {frame_idx})\n"
                                f"   Scoring:   Team {scoring_team+1} | "
                                f"Player #{scorer_j}\n"
                                f"   Conceding: Team {conceding_team+1}\n"
                                f"   Details:   {desc}\n"
                                f"   Evidence:  {len(evidence_summary)} frames "
                                f"around goal\n"
                            )
                else:
                    self._goal_streak = 0

            # ===== 4. Corner Kick Detection =====
            if ball_pos is not None:
                corners = [
                    (0, 0), (0, self.pitch_width),
                    (self.pitch_length, 0), (self.pitch_length, self.pitch_width)
                ]
                for cx, cy in corners:
                    dist_to_corner = np.sqrt((ball_pos[0] - cx)**2 + (ball_pos[1] - cy)**2)
                    if dist_to_corner < self.corner_area_radius:
                        events.append({
                            'frame_idx': frame_idx,
                            'timestamp': timestamp_str,
                            'timestamp_seconds': timestamp_sec,
                            'event_type': 'Corner Kick',
                            'players_involved': [],
                            'teams_involved': [],
                            'confidence': 0.70,
                            'description': f"Ball near corner flag ({dist_to_corner:.1f}m from corner)"
                        })
                        break  # Only one corner per frame

            # ===== 5. Shot on Target Detection =====
            if ball_pos is not None and frame_idx < len(ball_velocities) and ball_velocities[frame_idx] is not None:
                bvx, bvy = ball_velocities[frame_idx]
                ball_speed = np.sqrt(bvx**2 + bvy**2)

                if ball_speed > self.shot_velocity_thresh:
                    # Check if velocity direction points toward either goal
                    toward_right_goal = bvx > 0 and ball_pos[0] > self.pitch_length * 0.5
                    toward_left_goal = bvx < 0 and ball_pos[0] < self.pitch_length * 0.5

                    if toward_right_goal or toward_left_goal:
                        shooting_team = 0 if toward_right_goal else 1
                        events.append({
                            'frame_idx': frame_idx,
                            'timestamp': timestamp_str,
                            'timestamp_seconds': timestamp_sec,
                            'event_type': 'Shot on Target',
                            'players_involved': [],
                            'teams_involved': [shooting_team],
                            'confidence': round(min(0.90, 0.5 + ball_speed / 20), 2),
                            'description': f"Ball moving at {ball_speed:.1f} m/s toward goal (vx={bvx:.1f}, vy={bvy:.1f})"
                        })

            # ===== 6. Penalty Area Entry =====
            if ball_pos is not None:
                # Left penalty area: x < penalty_area_depth
                if ball_pos[0] < self.penalty_area_depth:
                    if self.penalty_area_y_min <= ball_pos[1] <= self.penalty_area_y_max:
                        # Check for attacking players (team 1 attacking left side)
                        attackers_in_box = [
                            p_id for p_id, p_pos in players_team_1
                            if p_pos[0] < self.penalty_area_depth
                            and self.penalty_area_y_min <= p_pos[1] <= self.penalty_area_y_max
                        ]
                        if attackers_in_box:
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Penalty Area Entry',
                                'players_involved': attackers_in_box,
                                'teams_involved': [1],
                                'confidence': 0.80,
                                'description': f"Team B attack in left penalty area with {len(attackers_in_box)} player(s)"
                            })

                # Right penalty area: x > pitch_length - penalty_area_depth
                if ball_pos[0] > self.pitch_length - self.penalty_area_depth:
                    if self.penalty_area_y_min <= ball_pos[1] <= self.penalty_area_y_max:
                        attackers_in_box = [
                            p_id for p_id, p_pos in players_team_0
                            if p_pos[0] > self.pitch_length - self.penalty_area_depth
                            and self.penalty_area_y_min <= p_pos[1] <= self.penalty_area_y_max
                        ]
                        if attackers_in_box:
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Penalty Area Entry',
                                'players_involved': attackers_in_box,
                                'teams_involved': [0],
                                'confidence': 0.80,
                                'description': f"Team A attack in right penalty area with {len(attackers_in_box)} player(s)"
                            })

            # ===== 7. Offside Detection =====
            if ball_pos is not None:
                offside_events = self._detect_offside(
                    frame_idx, ball_pos, players_team_0, players_team_1,
                    ball_velocities, timestamp_str, timestamp_sec
                )
                events.extend(offside_events)

            # ===== 8. Free Kick Detection =====
            if ball_pos is not None and frame_idx < len(ball_velocities) and ball_velocities[frame_idx] is not None:
                bvx, bvy = ball_velocities[frame_idx]
                ball_speed = np.sqrt(bvx**2 + bvy**2)

                # Ball nearly stationary after a foul
                if ball_speed < 0.5:
                    recent_foul = any(
                        abs(e['frame_idx'] - frame_idx) < int(self.fps * 5)
                        for e in events if e['event_type'] == 'Potential Foul'
                    )
                    if recent_foul:
                        # Check for player cluster around the ball (free kick setup)
                        players_near_ball = sum(
                            1 for _, p_pos in players_team_0 + players_team_1
                            if np.sqrt((p_pos[0] - ball_pos[0])**2 + (p_pos[1] - ball_pos[1])**2) < 5.0
                        )
                        if players_near_ball >= 3:
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Free Kick',
                                'players_involved': [],
                                'teams_involved': [],
                                'confidence': 0.65,
                                'description': f"Ball stationary with {players_near_ball} players nearby after foul"
                            })

            # Tag jerseys involved for new events in this frame
            if jersey_map:
                for evt in events:
                    if evt['frame_idx'] == frame_idx and 'jerseys_involved' not in evt:
                        evt['jerseys_involved'] = [jersey_map[pid] for pid in evt.get('players_involved', []) if pid in jersey_map]

        # Deduplicate consecutive frame detections of the same incident
        deduped_events = self._deduplicate_events(events)

        # Summary Log Table
        event_type_counts = {}
        for e in deduped_events:
            et = e['event_type']
            event_type_counts[et] = event_type_counts.get(et, 0) + 1

        print("\n" + "=" * 65)
        print("  [EventDetector] MATCH KEY EVENTS DETECTION SUMMARY")
        print("=" * 65)
        for e in deduped_events:
            prefix = "⚽" if e['event_type'] == 'Goal' else "🚨"
            print(f"  {prefix} [{e['timestamp']}] {e['event_type'].upper():<20} | {e['description']} (Conf: {e.get('confidence', 0.8):.2f})")
        print(f"\n  * Total Events Flagged: {len(deduped_events)} | Breakdown: {event_type_counts}")
        print("=" * 65 + "\n")

        return deduped_events

    def _detect_offside(
        self,
        frame_idx: int,
        ball_pos: Tuple[float, float],
        players_team_0: List[Tuple[int, Tuple[float, float]]],
        players_team_1: List[Tuple[int, Tuple[float, float]]],
        ball_velocities: List[Optional[Tuple[float, float]]],
        timestamp_str: str,
        timestamp_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Detects offside: attacker is ahead of 2nd-last defender when ball is played forward.

        Simple heuristic:
        - For Team 0 attacking right: check if any Team 0 player is beyond 2nd-last Team 1 defender
        - Ball must be moving forward (positive velocity in attacking direction)
        """
        events = []

        # Check ball velocity — only flag during forward pass moments
        if frame_idx >= len(ball_velocities) or ball_velocities[frame_idx] is None:
            return events

        bvx, bvy = ball_velocities[frame_idx]
        ball_speed = np.sqrt(bvx**2 + bvy**2)

        if ball_speed < 2.0:  # Ball not moving fast enough to be a pass
            return events

        # --- Team 0 attacking right (offside = beyond 2nd-last Team 1 defender) ---
        if bvx > 1.0 and players_team_1:
            # Sort Team 1 players by X position (highest X = closest to their goal)
            team_1_x_positions = sorted([p[1][0] for p in players_team_1], reverse=True)
            if len(team_1_x_positions) >= 2:
                second_last_defender_x = team_1_x_positions[1]  # 2nd-highest X

                for p_id, p_pos in players_team_0:
                    if p_pos[0] > second_last_defender_x + self.offside_margin:
                        # Player is offside relative to 2nd-last defender
                        if p_pos[0] > self.pitch_length * 0.5:  # Must be in opponent's half
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Offside',
                                'players_involved': [p_id],
                                'teams_involved': [0],
                                'confidence': 0.60,
                                'description': f"Player #{p_id} (Team A) beyond 2nd-last defender at {p_pos[0]:.1f}m (def line: {second_last_defender_x:.1f}m)"
                            })

        # --- Team 1 attacking left (offside = beyond 2nd-last Team 0 defender) ---
        if bvx < -1.0 and players_team_0:
            team_0_x_positions = sorted([p[1][0] for p in players_team_0])
            if len(team_0_x_positions) >= 2:
                second_last_defender_x = team_0_x_positions[1]  # 2nd-lowest X

                for p_id, p_pos in players_team_1:
                    if p_pos[0] < second_last_defender_x - self.offside_margin:
                        if p_pos[0] < self.pitch_length * 0.5:  # Must be in opponent's half
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Offside',
                                'players_involved': [p_id],
                                'teams_involved': [1],
                                'confidence': 0.60,
                                'description': f"Player #{p_id} (Team B) beyond 2nd-last defender at {p_pos[0]:.1f}m (def line: {second_last_defender_x:.1f}m)"
                            })

        return events

    def _compute_ball_velocities(
        self,
        ball_metric_per_frame: Optional[List[Optional[Tuple[float, float]]]]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Computes ball velocity (vx, vy) in m/s per frame from ball metric positions.
        """
        if not ball_metric_per_frame:
            return []

        velocities: List[Optional[Tuple[float, float]]] = [None]  # No velocity for first frame

        for i in range(1, len(ball_metric_per_frame)):
            prev = ball_metric_per_frame[i - 1]
            curr = ball_metric_per_frame[i]

            if prev is not None and curr is not None:
                vx = (curr[0] - prev[0]) * self.fps  # m/s
                vy = (curr[1] - prev[1]) * self.fps
                velocities.append((vx, vy))
            else:
                velocities.append(None)

        return velocities

    def _deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []

        deduped = []
        window_map = {
            'Potential Foul': 3.0,
            'Yellow Card Candidate': 5.0,
            'Goal': 30.0,
            'Corner Kick': 5.0,
            'Free Kick': 5.0,
            'Offside': 3.0,
            'Shot on Target': 3.0,
            'Penalty Area Entry': 2.0,
        }

        for event in events:
            is_dup = False
            window_sec = window_map.get(event['event_type'], 3.0)
            window_frames = int(self.fps * window_sec)

            for prev in deduped:
                if (
                    prev['event_type'] == event['event_type'] and
                    abs(prev['frame_idx'] - event['frame_idx']) <= window_frames
                ):
                    # For player-specific events, also check player overlap
                    if event['players_involved'] and prev['players_involved']:
                        if set(prev['players_involved']) & set(event['players_involved']):
                            is_dup = True
                            break
                    else:
                        is_dup = True
                        break

            if not is_dup:
                deduped.append(event)

        return deduped

    def _is_zoom_replay(
        self,
        current_tracks: List[Dict[str, Any]],
        prev_tracks: List[Dict[str, Any]],
        jersey_map: Dict[int, str],
        events: List[Dict[str, Any]]
    ) -> bool:
        """
        Detects if the frame is a zoomed-in replay of a recently involved player.
        """
        zoom_threshold = 250
        slow_mo_threshold = 10.0
        
        for track in current_tracks:
            bbox = track['bbox']
            h = bbox[3] - bbox[1]
            if h > zoom_threshold:
                track_id = track['track_id']
                
                # Check for slow motion (displacement < 10 pixels)
                prev_track = next((t for t in prev_tracks if t['track_id'] == track_id), None)
                if prev_track:
                    prev_bbox = prev_track['bbox']
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    px = (prev_bbox[0] + prev_bbox[2]) / 2
                    py = (prev_bbox[1] + prev_bbox[3]) / 2
                    displacement = np.sqrt((cx - px)**2 + (cy - py)**2)
                    
                    if displacement > slow_mo_threshold:
                        continue # Moving too fast to be slow-motion
                
                # It's zoomed in and slow-motion. Let's check jersey!
                if track_id in jersey_map:
                    jersey_num = jersey_map[track_id]
                    
                    # Look back through recent events
                    for event in reversed(events):
                        if 'jerseys_involved' in event and jersey_num in event['jerseys_involved']:
                            return True
        return False

    def point_in_goal_polygon(
        self,
        ball_pixel: Tuple[float, float],
        polygon: List[List[float]],
        expanded_px: float = 0.0
    ) -> bool:
        """
        Ray-Casting (Even-Odd) algorithm.
        Returns False (safe default) if inputs are invalid.

        `expanded_px`: pixel margin to add to each vertex outward from
        the polygon centroid. Use 8-12 px for airborne ball tolerance.
        """
        if (ball_pixel is None or not polygon or len(polygon) < 3):
            return False

        u, v = ball_pixel

        pts = np.array(polygon, dtype=float)
        if expanded_px > 0:
            centroid = pts.mean(axis=0)
            directions = pts - centroid
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            pts = pts + (directions / norms) * expanded_px

        n = len(pts)
        inside = False
        p1x, p1y = pts[0]
        for i in range(1, n + 1):
            p2x, p2y = pts[i % n]
            if (v > min(p1y, p2y)) and (v <= max(p1y, p2y)):
                if p2y != p1y:
                    x_int = (v - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if u < x_int:
                        inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _check_goal_hybrid(
        self,
        ball_pixel:      Optional[Tuple[float, float]],
        ground_pos:      Optional[Tuple[float, float]],
        goal_side:       str,
        goal_polygon:    Optional[List[List[float]]] = None,
        ball_velocity:   Optional[Tuple[float, float]] = None,
        ball_bbox:       Optional[Tuple[float,float,float,float]] = None,
        net_ripple_conf: float = 0.0,
        traj_cross_conf: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """
        5-signal weighted confidence fusion:

          Signal                  Weight   Source
          -----------------------------------------
          Ground footprint          0.25   homography (0.40 if ball_pixel is None)
          Pixel polygon             0.25   image-space ray-cast
          Velocity direction        0.15   physics
          Trajectory extrapolation  0.15   Kalman linear fit
          Net ripple                0.20   optical flow on net ROI

        Goal confirmed when total_conf >= THRESHOLD (0.55 default).
        """
        THRESHOLD = 0.55

        if ground_pos is None:
            return False, 0.0, ""

        X_g, Y_g = ground_pos

        # ── ABSOLUTE SPATIAL BOUNDARY GUARD ──────────────────────────
        # A goal is physically impossible if the ball is in midfield (16.5m < X_g < 88.5m)
        if goal_side == 'left' and X_g > 16.5:
            return False, 0.0, ""
        if goal_side == 'right' and X_g < 88.5:
            return False, 0.0, ""

        signals: Dict[str, float] = {}
        reasons:  List[str]       = []

        # ── Estimate ball height ─────────────────────────────
        z_est = (self.ball_buffer.estimate_height_m()
                 if hasattr(self, 'ball_buffer') else 0.0)
        is_airborne = z_est > 0.25
        airborne_ok = z_est <= (2.44 + 0.30)   # within crossbar height

        # ── SIGNAL 1: Ground footprint ───────────────────────
        is_within_y = self.goal_mouth_y_min <= Y_g <= self.goal_mouth_y_max

        if goal_side == 'right':
            is_past_line = X_g >= (self.pitch_length - self.goal_line_thresh)
        else:
            is_past_line = X_g <= self.goal_line_thresh

        if is_past_line and is_within_y:
            if is_airborne:
                w = 0.15
            elif ball_pixel is None:
                w = 0.40
            else:
                w = 0.25
            signals['ground'] = w
            reasons.append(f"ground({X_g:.1f}m,{Y_g:.1f}m)")

        # ── SIGNAL 2: Pixel polygon (leading-edge check) ─────
        if ball_pixel is not None and goal_polygon is not None and airborne_ok:
            if ball_bbox is not None:
                x1, y1, x2, y2 = ball_bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if goal_side == 'right':
                    leading_px = (x2, cy)
                else:
                    leading_px = (x1, cy)
            else:
                leading_px = ball_pixel

            airborne_expand = min(z_est * 30, 60)

            if is_airborne and airborne_expand > 0:
                poly_arr = np.array(goal_polygon, dtype=float)
                top_indices = [2, 3]
                for ti in top_indices:
                    poly_arr[ti][1] -= airborne_expand
                expanded_polygon = poly_arr.tolist()
            else:
                expanded_polygon = goal_polygon

            in_poly = self.point_in_goal_polygon(
                leading_px, expanded_polygon, expanded_px=8.0
            )
            if in_poly:
                signals['pixel'] = 0.25
                reasons.append("pixel_polygon")

        # ── SIGNAL 3: Velocity direction ─────────────────────
        vel_veto = False
        if ball_velocity is not None:
            bvx, bvy = ball_velocity
            if goal_side == 'right':
                if bvx > -3.0:
                    signals['velocity'] = 0.15
                    reasons.append("vel_ok")
                else:
                    vel_veto = True
            else:
                if bvx < 3.0:
                    signals['velocity'] = 0.15
                    reasons.append("vel_ok")
                else:
                    vel_veto = True
        else:
            signals['velocity'] = 0.08

        # ── SIGNAL 4: Trajectory extrapolation ───────────────
        if traj_cross_conf > 0.0:
            signals['trajectory'] = traj_cross_conf * 0.15
            reasons.append(f"traj({traj_cross_conf:.2f})")

        # ── SIGNAL 5: Net ripple ─────────────────────────────
        if net_ripple_conf > 0.0:
            signals['net_ripple'] = net_ripple_conf * 0.20
            reasons.append(f"net_ripple({net_ripple_conf:.2f})")

        # ── Fuse signals ─────────────────────────────────────
        total = sum(signals.values())

        if vel_veto and net_ripple_conf < 0.4:
            total *= 0.25
            reasons.append("VEL_VETO")

        if (net_ripple_conf > 0.7 and
                any(v > 0.4 for k, v in signals.items() if k != 'net_ripple')):
            total = max(total, 0.82)
            reasons.append("RIPPLE_CONFIRM")

        has_net_evidence = 'ground' in signals or 'pixel' in signals
        confirmed = total >= THRESHOLD and has_net_evidence
        side_str  = goal_side
        desc      = (f"GoalFusion({side_str}): "
                     f"conf={total:.2f} [{' | '.join(reasons)}] "
                     f"Z={z_est:.1f}m")

        return confirmed, round(min(total, 1.0), 2), desc

