"""
Stage G: Tactical Analysis — Formation Detection, Passing Network & Defensive Metrics.

Automatically infers high-level tactical insights from tracked player positions:

    | Metric                | Method                                                |
    |----------------------|-------------------------------------------------------|
    | Formation Detection  | K-Means clustering of player positions into lines      |
    | Defensive Line Height| Average Y position of back 4 defenders                 |
    | Pressing Intensity   | Count players within 5m of ball carrier                |
    | Attacking Zones      | Ball time in defensive/mid/attacking thirds            |
    | Passing Network      | From touch/possession transitions between players      |
    | Compactness          | Convex hull area of team player positions               |
    | Width/Depth          | Max horizontal & vertical spread of team               |

Uses only NumPy + scikit-learn clustering (already in requirements.txt).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Standard formation templates for matching
FORMATION_TEMPLATES = {
    '4-4-2': [1, 4, 4, 2],
    '4-3-3': [1, 4, 3, 3],
    '3-5-2': [1, 3, 5, 2],
    '4-2-3-1': [1, 4, 2, 3, 1],
    '4-1-4-1': [1, 4, 1, 4, 1],
    '3-4-3': [1, 3, 4, 3],
    '5-3-2': [1, 5, 3, 2],
    '5-4-1': [1, 5, 4, 1],
    '4-5-1': [1, 4, 5, 1],
    '4-4-1-1': [1, 4, 4, 1, 1],
}


class TacticalAnalyzer:
    """
    Computes tactical metrics from tracked player metric positions per frame.

    Expects positions in pitch metric coordinates (0-105m x 0-68m).
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        pressing_radius_m: float = 5.0,
        possession_radius_m: float = 2.0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.pressing_radius = pressing_radius_m
        self.possession_radius = possession_radius_m
        self.third_length = pitch_length / 3.0

    def analyze(
        self,
        tracks_per_frame: List[List[Dict[str, Any]]],
        metric_positions_per_frame: List[Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, int],
        ball_metric_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None
    ) -> Dict[str, Any]:
        """
        Runs full tactical analysis across all frames.

        Returns:
            Dict containing formation, defensive metrics, pressing, attacking zones,
            passing network, compactness, and width/depth stats per team.
        """
        n_frames = len(metric_positions_per_frame)

        # Accumulate per-team positions across all frames
        team_positions_all: Dict[int, List[List[Tuple[float, float]]]] = {0: [], 1: []}
        ball_zone_counts = {'defensive': 0, 'midfield': 0, 'attacking': 0, 'total': 0}
        pressing_intensities: List[float] = []
        defensive_line_heights: Dict[int, List[float]] = {0: [], 1: []}
        compactness: Dict[int, List[float]] = {0: [], 1: []}
        width_depth: Dict[int, Dict[str, List[float]]] = {
            0: {'width': [], 'depth': []},
            1: {'width': [], 'depth': []}
        }

        # Passing network: who touches ball after whom
        possession_sequence: List[Tuple[int, int]] = []  # (frame_idx, track_id)
        touch_counts: Dict[int, int] = defaultdict(int)
        pass_counts: Dict[Tuple[int, int], int] = defaultdict(int)  # (from_id, to_id) → count

        last_possessor = None

        for frame_idx in range(n_frames):
            pos_dict = metric_positions_per_frame[frame_idx]
            tracks = tracks_per_frame[frame_idx]

            # Separate players by team
            team_frame_pos: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
            for track in tracks:
                t_id = track['track_id']
                if track['class_id'] not in (0, 1):
                    continue
                if t_id not in pos_dict:
                    continue
                team_id = team_assignments.get(t_id, 0)
                team_frame_pos[team_id].append(pos_dict[t_id])

            for team_id in [0, 1]:
                if team_frame_pos[team_id]:
                    team_positions_all[team_id].append(team_frame_pos[team_id])

                    positions_arr = np.array(team_frame_pos[team_id])

                    # Defensive line height (back 4 — lowest Y positions on their attacking axis)
                    if len(positions_arr) >= 4:
                        # Sort by X (attacking direction) and take the 4 lowest
                        sorted_x = np.sort(positions_arr[:, 0])
                        def_line = np.mean(sorted_x[:4])
                        defensive_line_heights[team_id].append(def_line)

                    # Compactness (convex hull area)
                    if len(positions_arr) >= 3:
                        try:
                            from scipy.spatial import ConvexHull
                            hull = ConvexHull(positions_arr)
                            compactness[team_id].append(hull.volume)  # 2D area
                        except Exception:
                            pass

                    # Width & Depth
                    if len(positions_arr) >= 2:
                        w = float(np.max(positions_arr[:, 1]) - np.min(positions_arr[:, 1]))
                        d = float(np.max(positions_arr[:, 0]) - np.min(positions_arr[:, 0]))
                        width_depth[team_id]['width'].append(w)
                        width_depth[team_id]['depth'].append(d)

            # Ball zone tracking
            ball_pos = None
            if ball_metric_per_frame and frame_idx < len(ball_metric_per_frame):
                ball_pos = ball_metric_per_frame[frame_idx]

            if ball_pos is not None:
                ball_zone_counts['total'] += 1
                bx = ball_pos[0]
                if bx < self.third_length:
                    ball_zone_counts['defensive'] += 1
                elif bx < 2 * self.third_length:
                    ball_zone_counts['midfield'] += 1
                else:
                    ball_zone_counts['attacking'] += 1

                # Pressing intensity: count players from opposing team within radius of ball
                for track in tracks:
                    t_id = track['track_id']
                    if track['class_id'] != 0 or t_id not in pos_dict:
                        continue

                pressing_count = 0
                for track in tracks:
                    t_id = track['track_id']
                    if track['class_id'] != 0 or t_id not in pos_dict:
                        continue
                    p_pos = pos_dict[t_id]
                    dist = np.sqrt((p_pos[0] - ball_pos[0]) ** 2 + (p_pos[1] - ball_pos[1]) ** 2)
                    if dist < self.pressing_radius:
                        pressing_count += 1
                pressing_intensities.append(pressing_count)

                # Possession tracking (for passing network)
                closest_player = None
                min_dist = float('inf')
                for track in tracks:
                    t_id = track['track_id']
                    if track['class_id'] != 0 or t_id not in pos_dict:
                        continue
                    p_pos = pos_dict[t_id]
                    dist = np.sqrt((p_pos[0] - ball_pos[0]) ** 2 + (p_pos[1] - ball_pos[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = t_id

                if closest_player is not None and min_dist < self.possession_radius:
                    touch_counts[closest_player] += 1
                    if last_possessor is not None and last_possessor != closest_player:
                        # Same team? → pass. Different team? → interception
                        if team_assignments.get(last_possessor, 0) == team_assignments.get(closest_player, 0):
                            pass_counts[(last_possessor, closest_player)] += 1
                    last_possessor = closest_player

        # ----- Formation Detection -----
        formations = {}
        for team_id in [0, 1]:
            formation = self._detect_formation(team_positions_all[team_id])
            formations[team_id] = formation

        # ----- Compile Results -----
        # Attacking zones as percentages
        total_ball_frames = max(ball_zone_counts['total'], 1)
        attacking_zones = {
            'defensive_third_pct': round(ball_zone_counts['defensive'] / total_ball_frames * 100, 1),
            'midfield_third_pct': round(ball_zone_counts['midfield'] / total_ball_frames * 100, 1),
            'attacking_third_pct': round(ball_zone_counts['attacking'] / total_ball_frames * 100, 1)
        }

        # Passing network: top connections
        passing_network = []
        for (from_id, to_id), count in sorted(pass_counts.items(), key=lambda x: -x[1])[:20]:
            passing_network.append({
                'from_player': from_id,
                'to_player': to_id,
                'pass_count': count,
                'team': team_assignments.get(from_id, 0)
            })

        # Touch counts
        touch_summary = {pid: cnt for pid, cnt in sorted(touch_counts.items(), key=lambda x: -x[1])}

        results = {
            'formations': {
                f'team_{tid}': formations[tid] for tid in [0, 1]
            },
            'defensive_line_height': {
                f'team_{tid}': round(float(np.mean(defensive_line_heights[tid])), 1)
                if defensive_line_heights[tid] else 0.0
                for tid in [0, 1]
            },
            'pressing_intensity': {
                'avg_players_near_ball': round(float(np.mean(pressing_intensities)), 2)
                if pressing_intensities else 0.0,
                'max_players_near_ball': max(pressing_intensities) if pressing_intensities else 0
            },
            'attacking_zones': attacking_zones,
            'passing_network': passing_network,
            'touch_counts': touch_summary,
            'compactness': {
                f'team_{tid}': round(float(np.mean(compactness[tid])), 1)
                if compactness[tid] else 0.0
                for tid in [0, 1]
            },
            'width_depth': {
                f'team_{tid}': {
                    'avg_width_m': round(float(np.mean(width_depth[tid]['width'])), 1)
                    if width_depth[tid]['width'] else 0.0,
                    'avg_depth_m': round(float(np.mean(width_depth[tid]['depth'])), 1)
                    if width_depth[tid]['depth'] else 0.0
                }
                for tid in [0, 1]
            }
        }

        print(f"[TacticalAnalyzer] Formation Team A: {formations[0]}, Team B: {formations[1]}")
        print(f"[TacticalAnalyzer] Attacking zones: {attacking_zones}")
        print(f"[TacticalAnalyzer] Passes detected: {len(passing_network)}")

        return results

    def _detect_formation(
        self,
        team_positions_per_frame: List[List[Tuple[float, float]]]
    ) -> str:
        """
        Detects team formation by clustering average player positions into horizontal lines.

        Uses K-Means to cluster the X-coordinate (attacking axis) of average positions
        and counts players per line cluster to match against standard formation templates.

        Returns:
            Formation string like '4-4-2' or 'Unknown' if insufficient data.
        """
        if not team_positions_per_frame or not SKLEARN_AVAILABLE:
            return 'Unknown'

        # Build average position per player slot
        # Flatten all frame positions, keeping slot order within each frame
        all_positions = []
        for frame_positions in team_positions_per_frame:
            if len(frame_positions) >= 8:  # Need at least 8 outfield players
                all_positions.extend(frame_positions)

        if len(all_positions) < 20:
            return 'Unknown'

        positions_arr = np.array(all_positions)

        # Cluster by X-coordinate (attacking axis) to find formation lines
        x_coords = positions_arr[:, 0].reshape(-1, 1)

        best_formation = 'Unknown'
        best_score = float('inf')

        for formation_name, template in FORMATION_TEMPLATES.items():
            n_lines = len(template)
            if n_lines < 2:
                continue

            try:
                kmeans = KMeans(n_clusters=n_lines, n_init=10, random_state=42)
                labels = kmeans.fit_predict(x_coords)

                # Count players per cluster, sorted by cluster center X position
                centers = kmeans.cluster_centers_.flatten()
                sorted_indices = np.argsort(centers)
                counts_per_line = []
                for cluster_idx in sorted_indices:
                    counts_per_line.append(int(np.sum(labels == cluster_idx)))

                # Normalize counts to sum to 11
                total = sum(counts_per_line)
                if total == 0:
                    continue
                normalized = [round(c / total * 11) for c in counts_per_line]

                # Compute match score (lower is better)
                score = sum(abs(a - b) for a, b in zip(normalized, template))

                if score < best_score:
                    best_score = score
                    best_formation = formation_name

            except Exception:
                continue

        return best_formation

    def generate_tactical_report_text(self, tactical_data: Dict[str, Any]) -> str:
        """
        Generates a human-readable tactical analysis text summary.

        Args:
            tactical_data: Output from self.analyze()

        Returns:
            Multi-line string with tactical insights.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TACTICAL ANALYSIS REPORT")
        lines.append("=" * 60)

        # Formations
        for team_key in ['team_0', 'team_1']:
            team_label = 'Team A' if '0' in team_key else 'Team B'
            formation = tactical_data['formations'].get(team_key, 'Unknown')
            lines.append(f"\n{team_label} Formation: {formation}")

        # Defensive line
        for team_key in ['team_0', 'team_1']:
            team_label = 'Team A' if '0' in team_key else 'Team B'
            def_line = tactical_data['defensive_line_height'].get(team_key, 0.0)
            lines.append(f"{team_label} Defensive Line Height: {def_line}m from own goal")

        # Pressing
        pressing = tactical_data.get('pressing_intensity', {})
        lines.append(f"\nPressing Intensity: Avg {pressing.get('avg_players_near_ball', 0):.1f} players near ball")
        lines.append(f"  Max simultaneous press: {pressing.get('max_players_near_ball', 0)} players")

        # Attacking zones
        zones = tactical_data.get('attacking_zones', {})
        lines.append(f"\nBall Position by Third:")
        lines.append(f"  Defensive: {zones.get('defensive_third_pct', 0)}%")
        lines.append(f"  Midfield:  {zones.get('midfield_third_pct', 0)}%")
        lines.append(f"  Attacking: {zones.get('attacking_third_pct', 0)}%")

        # Compactness
        for team_key in ['team_0', 'team_1']:
            team_label = 'Team A' if '0' in team_key else 'Team B'
            comp = tactical_data['compactness'].get(team_key, 0.0)
            wd = tactical_data['width_depth'].get(team_key, {})
            lines.append(f"\n{team_label} Shape:")
            lines.append(f"  Compactness (hull area): {comp} m²")
            lines.append(f"  Average width: {wd.get('avg_width_m', 0)}m")
            lines.append(f"  Average depth: {wd.get('avg_depth_m', 0)}m")

        # Passing network
        top_passes = tactical_data.get('passing_network', [])[:5]
        if top_passes:
            lines.append(f"\nTop Passing Connections:")
            for p in top_passes:
                lines.append(f"  Player #{p['from_player']} → Player #{p['to_player']}: {p['pass_count']} passes")

        # Touch counts
        touches = tactical_data.get('touch_counts', {})
        top_touches = list(touches.items())[:5]
        if top_touches:
            lines.append(f"\nMost Touches:")
            for pid, cnt in top_touches:
                lines.append(f"  Player #{pid}: {cnt} touches")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
