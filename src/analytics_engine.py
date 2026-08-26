import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

class AnalyticsEngine:
    """
    Stage 8 (Enhanced): Computes comprehensive analytical metrics:
    - Speed (m/s and km/h) & total distance covered (meters)
    - Ball possession percentage per team
    - Team & Player position heatmaps
    - Sprint count (>25 km/h threshold)
    - Acceleration and deceleration profiles
    - Pass count and touch count per player
    - Pressure map (players near ball carrier)
    - Running direction (forward/backward/lateral)
    - Per-team aggregated heatmap data
    """
    def __init__(
        self,
        fps: float = 25.0,
        possession_proximity_thresh: float = 2.0,
        smoothing_window: int = 7,
        sprint_speed_thresh_kmh: float = 25.0,
        high_accel_thresh: float = 3.0
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.possession_thresh = possession_proximity_thresh
        self.smoothing_window = smoothing_window
        self.sprint_speed_thresh_ms = sprint_speed_thresh_kmh / 3.6  # Convert to m/s
        self.high_accel_thresh = high_accel_thresh  # m/s² threshold for "high acceleration"

    def compute_analytics(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Processes tracked metric positions across frames.
        Returns dictionary of player stats, team possession metrics, position heatmaps,
        sprint counts, acceleration data, and per-frame speed lookups.
        """
        player_trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        player_frame_indices: Dict[int, List[int]] = defaultdict(list)
        ball_trajectories: List[Tuple[int, Tuple[float, float]]] = []

        # 1. Organize trajectories by ID
        for frame_idx, pos_dict in enumerate(metric_positions_per_frame):
            frame_tracks = tracks_per_frame[frame_idx]
            for track in frame_tracks:
                track_id = track['track_id']
                cls_id = track['class_id']

                if track_id in pos_dict:
                    pos = pos_dict[track_id]
                    if cls_id == 3: # Ball
                        ball_trajectories.append((frame_idx, pos))
                    else:
                        player_trajectories[track_id].append(pos)
                        player_frame_indices[track_id].append(frame_idx)

        # 2. Compute Distance, Speed, Sprints, Acceleration per Player
        player_stats: Dict[int, Dict[str, Any]] = {}
        player_speeds_per_frame_dict: Dict[int, Dict[int, float]] = defaultdict(dict)  # {frame_idx: {track_id: speed_kmh}}

        for player_id, positions in player_trajectories.items():
            if len(positions) < 2:
                continue

            # Smooth positions using moving average
            smoothed_positions = self._smooth_trajectory(positions)
            frame_indices = player_frame_indices[player_id]
            
            total_distance = 0.0
            speeds_m_s = []
            accelerations = []
            sprint_count = 0
            in_sprint = False
            sprint_frames = 0
            direction_vectors = []
            touch_count = 0

            for i in range(1, len(smoothed_positions)):
                dx = smoothed_positions[i][0] - smoothed_positions[i-1][0]
                dy = smoothed_positions[i][1] - smoothed_positions[i-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                total_distance += dist

                speed = dist / self.dt
                # Cap speed spikes caused by tracking noise or ID switches at max realistic sprint speed (38 km/h = 10.55 m/s)
                speed = min(speed, 10.55)
                speeds_m_s.append(speed)

                # Store per-frame speed for action recognition
                if i < len(frame_indices):
                    player_speeds_per_frame_dict[frame_indices[i]][player_id] = speed * 3.6  # km/h

                # Sprint detection (>25 km/h = 6.94 m/s)
                if speed >= self.sprint_speed_thresh_ms:
                    if not in_sprint:
                        sprint_count += 1
                        in_sprint = True
                    sprint_frames += 1
                else:
                    in_sprint = False

                # Acceleration (m/s²) — change in speed over time
                if len(speeds_m_s) >= 2:
                    accel = (speeds_m_s[-1] - speeds_m_s[-2]) / self.dt
                    accelerations.append(accel)

                # Running direction vector (normalized)
                if dist > 0.01:  # Avoid division by zero for stationary
                    direction_vectors.append((dx / dist, dy / dist))

            avg_speed_m_s = float(np.mean(speeds_m_s)) if speeds_m_s else 0.0
            max_speed_m_s = float(np.max(speeds_m_s)) if speeds_m_s else 0.0
            avg_accel = float(np.mean(np.abs(accelerations))) if accelerations else 0.0
            max_accel = float(np.max(accelerations)) if accelerations else 0.0
            max_decel = float(np.min(accelerations)) if accelerations else 0.0
            high_accel_count = sum(1 for a in accelerations if abs(a) > self.high_accel_thresh)
            sprint_time_sec = sprint_frames * self.dt

            # Average running direction
            avg_direction = (0.0, 0.0)
            if direction_vectors:
                avg_dx = float(np.mean([d[0] for d in direction_vectors]))
                avg_dy = float(np.mean([d[1] for d in direction_vectors]))
                avg_direction = (round(avg_dx, 3), round(avg_dy, 3))

            team_id = team_assignments.get(player_id, 0)
            player_stats[player_id] = {
                'player_id': player_id,
                'team_id': team_id,
                'total_distance_m': round(total_distance, 2),
                'avg_speed_m_s': round(avg_speed_m_s, 2),
                'avg_speed_km_h': round(avg_speed_m_s * 3.6, 2),
                'max_speed_km_h': round(max_speed_m_s * 3.6, 2),
                'sprint_count': sprint_count,
                'sprint_time_sec': round(sprint_time_sec, 2),
                'avg_acceleration_ms2': round(avg_accel, 3),
                'max_acceleration_ms2': round(max_accel, 3),
                'max_deceleration_ms2': round(max_decel, 3),
                'high_intensity_changes': high_accel_count,
                'avg_running_direction': avg_direction,
                'positions': smoothed_positions
            }

        # 3. Compute Ball Possession Percentage
        possession_stats = self._compute_possession(
            tracks_per_frame,
            metric_positions_per_frame,
            team_assignments
        )

        # 4. Compute per-frame speed dict for action recognition
        speeds_per_frame_list: List[Dict[int, float]] = []
        for frame_idx in range(len(metric_positions_per_frame)):
            speeds_per_frame_list.append(
                player_speeds_per_frame_dict.get(frame_idx, {})
            )

        # 5. Compute ball metric position per frame (for tactical analysis)
        ball_metric_per_frame: List[Optional[Tuple[float, float]]] = []
        ball_dict: Dict[int, Tuple[float, float]] = {}
        for frame_idx, pos in ball_trajectories:
            ball_dict[frame_idx] = pos
        for frame_idx in range(len(metric_positions_per_frame)):
            ball_metric_per_frame.append(ball_dict.get(frame_idx))

        # 6. Compute touch counts and pass counts from possession transitions
        touch_counts, pass_count_per_player = self._compute_touch_pass_counts(
            tracks_per_frame, metric_positions_per_frame, team_assignments
        )

        # Merge touch/pass counts into player stats
        for pid in player_stats:
            player_stats[pid]['touch_count'] = touch_counts.get(pid, 0)
            player_stats[pid]['pass_count'] = pass_count_per_player.get(pid, 0)

        # 7. Compute per-team heatmap positions
        team_heatmaps: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
        for pid, stats in player_stats.items():
            tid = stats['team_id']
            team_heatmaps[tid].extend(stats.get('positions', []))

        # 8. Compute pressure map (per-frame count of opposing players near ball carrier)
        pressure_data = self._compute_pressure_map(
            tracks_per_frame, metric_positions_per_frame, team_assignments
        )

        return {
            'player_stats': player_stats,
            'possession_stats': possession_stats,
            'ball_trajectories': ball_trajectories,
            'speeds_per_frame': speeds_per_frame_list,
            'ball_metric_per_frame': ball_metric_per_frame,
            'touch_counts': touch_counts,
            'pass_counts': pass_count_per_player,
            'team_heatmaps': team_heatmaps,
            'pressure_data': pressure_data
        }

    def _smooth_trajectory(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Applies moving average filter over coordinate sequence.
        """
        if len(positions) < self.smoothing_window:
            return positions

        if PANDAS_AVAILABLE:
            df = pd.DataFrame(positions, columns=['x', 'y'])
            df_smooth = df.rolling(window=self.smoothing_window, min_periods=1, center=True).mean()
            return list(zip(df_smooth['x'], df_smooth['y']))
        else:
            # Fallback simple moving average with numpy
            arr = np.array(positions)
            kernel = np.ones(self.smoothing_window) / self.smoothing_window
            smooth_x = np.convolve(arr[:, 0], kernel, mode='same')
            smooth_y = np.convolve(arr[:, 1], kernel, mode='same')
            return list(zip(smooth_x, smooth_y))

    def _compute_possession(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int]
    ) -> Dict[str, Any]:
        """
        Determines which player/team is closest to ball per frame within possession_thresh.
        """
        team_possession_counts = {0: 0, 1: 0, 'none': 0}
        total_frames = len(metric_positions_per_frame)

        for frame_idx, pos_dict in enumerate(metric_positions_per_frame):
            tracks = tracks_per_frame[frame_idx]
            ball_pos = None

            # Find ball position
            for track in tracks:
                if track['class_id'] == 3 and track['track_id'] in pos_dict:
                    ball_pos = pos_dict[track['track_id']]
                    break

            if ball_pos is None:
                team_possession_counts['none'] += 1
                continue

            min_dist = float('inf')
            closest_player_team = None

            for track in tracks:
                if track['class_id'] == 0 and track['track_id'] in pos_dict:
                    player_pos = pos_dict[track['track_id']]
                    dist = np.sqrt((ball_pos[0] - player_pos[0])**2 + (ball_pos[1] - player_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player_team = team_assignments.get(track['track_id'], 0)

            if min_dist <= self.possession_thresh and closest_player_team is not None:
                team_possession_counts[closest_player_team] += 1
            else:
                team_possession_counts['none'] += 1

        claimed_frames = team_possession_counts[0] + team_possession_counts[1]
        if claimed_frames > 0:
            team_a_pct = round((team_possession_counts[0] / claimed_frames) * 100, 1)
            team_b_pct = round((team_possession_counts[1] / claimed_frames) * 100, 1)
        else:
            team_a_pct, team_b_pct = 50.0, 50.0

        return {
            'team_a_possession_pct': team_a_pct,
            'team_b_possession_pct': team_b_pct,
            'raw_frame_counts': team_possession_counts
        }

    def _compute_touch_pass_counts(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Computes touch count and pass count per player by tracking ball proximity transitions.

        A "touch" occurs when a player is closest to the ball within possession_thresh.
        A "pass" occurs when possession transitions from player A to player B on the same team.

        Returns:
            (touch_counts: {player_id: count}, pass_counts: {player_id: count})
        """
        touch_counts: Dict[int, int] = defaultdict(int)
        pass_counts: Dict[int, int] = defaultdict(int)  # Passes initiated by player
        last_possessor = None

        for frame_idx, pos_dict in enumerate(metric_positions_per_frame):
            tracks = tracks_per_frame[frame_idx]

            # Find ball
            ball_pos = None
            for track in tracks:
                if track['class_id'] == 3 and track['track_id'] in pos_dict:
                    ball_pos = pos_dict[track['track_id']]
                    break

            if ball_pos is None:
                continue

            # Find closest player to ball
            min_dist = float('inf')
            closest_player = None
            for track in tracks:
                if track['class_id'] != 0 or track['track_id'] not in pos_dict:
                    continue
                p_pos = pos_dict[track['track_id']]
                dist = np.sqrt((ball_pos[0] - p_pos[0]) ** 2 + (ball_pos[1] - p_pos[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = track['track_id']

            if closest_player is not None and min_dist <= self.possession_thresh:
                touch_counts[closest_player] += 1

                # Check for pass (possession transfer within same team)
                if last_possessor is not None and last_possessor != closest_player:
                    if team_assignments.get(last_possessor, 0) == team_assignments.get(closest_player, 0):
                        pass_counts[last_possessor] += 1

                last_possessor = closest_player

        return dict(touch_counts), dict(pass_counts)

    def _compute_pressure_map(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int],
        pressure_radius: float = 5.0
    ) -> Dict[str, Any]:
        """
        Computes pressure data: how many opposing players are near the ball carrier per frame.

        Returns:
            Dict with average pressure, max pressure, and per-frame pressure counts.
        """
        pressure_counts = []

        for frame_idx, pos_dict in enumerate(metric_positions_per_frame):
            tracks = tracks_per_frame[frame_idx]

            ball_pos = None
            for track in tracks:
                if track['class_id'] == 3 and track['track_id'] in pos_dict:
                    ball_pos = pos_dict[track['track_id']]
                    break

            if ball_pos is None:
                continue

            # Find ball carrier team
            min_dist = float('inf')
            carrier_team = None
            for track in tracks:
                if track['class_id'] != 0 or track['track_id'] not in pos_dict:
                    continue
                p_pos = pos_dict[track['track_id']]
                dist = np.sqrt((ball_pos[0] - p_pos[0]) ** 2 + (ball_pos[1] - p_pos[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    carrier_team = team_assignments.get(track['track_id'], 0)

            if carrier_team is None or min_dist > self.possession_thresh:
                continue

            # Count opposing players within pressure_radius of ball
            opposing_near = 0
            for track in tracks:
                if track['class_id'] != 0 or track['track_id'] not in pos_dict:
                    continue
                t_team = team_assignments.get(track['track_id'], 0)
                if t_team == carrier_team:
                    continue
                p_pos = pos_dict[track['track_id']]
                dist = np.sqrt((ball_pos[0] - p_pos[0]) ** 2 + (ball_pos[1] - p_pos[1]) ** 2)
                if dist < pressure_radius:
                    opposing_near += 1

            pressure_counts.append(opposing_near)

        return {
            'avg_pressure': round(float(np.mean(pressure_counts)), 2) if pressure_counts else 0.0,
            'max_pressure': max(pressure_counts) if pressure_counts else 0,
            'pressure_frames': len(pressure_counts)
        }
