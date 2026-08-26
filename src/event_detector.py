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

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


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
        consecutive_goal_frames: int = 2
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
            [30, 480], [110, 470], [110, 230], [30, 240]
        ]
        self.right_goal_polygon = right_goal_polygon or [
            [1170, 470], [1250, 480], [1250, 240], [1170, 230]
        ]
        self.consecutive_goal_frames = consecutive_goal_frames
        self._goal_streak = 0

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
        ball_pixels_per_frame: Optional[List[Optional[Tuple[float, float]]]] = None
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

            # ===== 3. 2.5D Hybrid Goal Detection (Footprint + Pixel Polygon + Physics Fusion) =====
            if ball_pos is not None:
                curr_ball_px = ball_pixels_per_frame[frame_idx] if (ball_pixels_per_frame and frame_idx < len(ball_pixels_per_frame)) else None
                
                # Check ball velocity vector direction (must be moving toward goal at shot speed)
                ball_vel = ball_velocities[frame_idx] if (frame_idx < len(ball_velocities) and ball_velocities[frame_idx] is not None) else (0.0, 0.0)
                bvx, bvy = ball_vel
                speed_ms = np.sqrt(bvx**2 + bvy**2)
                
                # Check Left Goal (Team B attacks left, vx <= 0)
                is_goal_left, conf_left, desc_left = self._check_goal_hybrid(
                    ball_pixel=curr_ball_px,
                    ground_pos=ball_pos,
                    goal_side='left',
                    goal_polygon=self.left_goal_polygon,
                    ball_velocity=ball_vel
                )
                
                # Check Right Goal (Team A attacks right, vx >= 0)
                is_goal_right, conf_right, desc_right = self._check_goal_hybrid(
                    ball_pixel=curr_ball_px,
                    ground_pos=ball_pos,
                    goal_side='right',
                    goal_polygon=self.right_goal_polygon,
                    ball_velocity=ball_vel
                )

                if is_goal_left or is_goal_right:
                    self._goal_streak += 1
                    conf = conf_left if is_goal_left else conf_right
                    desc = desc_left if is_goal_left else desc_right
                    
                    if self._goal_streak >= self.consecutive_goal_frames:
                        # Find shooter (closest player to ball trajectory during shot)
                        all_players = players_team_0 + players_team_1
                        scorer_id = None
                        min_dist_to_ball = 999.0
                        for p_id, p_pos in all_players:
                            dist_p = np.sqrt((p_pos[0] - ball_pos[0])**2 + (p_pos[1] - ball_pos[1])**2)
                            if dist_p < min_dist_to_ball:
                                min_dist_to_ball = dist_p
                                scorer_id = p_id
                        
                        # Dynamically assign scoring team from the shooter's team
                        if scorer_id is not None:
                            scoring_team = team_assignments.get(scorer_id, 1 if is_goal_left else 0)
                        else:
                            scoring_team = 1 if is_goal_left else 0
                        
                        scorer_desc = f" | Scorer: Player #{jersey_map.get(scorer_id, scorer_id)}" if scorer_id is not None else ""
                        
                        # Prevent duplicate goals from slow-motion replays or camera angle cuts
                        # A new goal requires at least 20 seconds cooldown OR kickoff reset
                        recent_goal = any(
                            abs(e['frame_idx'] - frame_idx) < int(self.fps * 20.0) and e['event_type'] == 'Goal'
                            for e in events
                        )
                        if not recent_goal:
                            events.append({
                                'frame_idx': frame_idx,
                                'timestamp': timestamp_str,
                                'timestamp_seconds': timestamp_sec,
                                'event_type': 'Goal',
                                'players_involved': [scorer_id] if scorer_id else [],
                                'teams_involved': [scoring_team],
                                'confidence': conf,
                                'description': desc + scorer_desc
                            })
                            team_label = "Team White / Team B" if scoring_team == 1 else "Team Red / Team A"
                            print(f"\n⚽ [EVENT DETECTOR] GOAL CONFIRMED at {timestamp_str} (Frame {frame_idx})!")
                            print(f"   -> Scoring Team: {team_label} | Scorer: Player #{jersey_map.get(scorer_id, scorer_id) if scorer_id else 'Unknown'} | {desc}\n")
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
        polygon: List[List[float]]
    ) -> bool:
        """
        Ray-Casting algorithm to check if a 2D ball pixel (u, v) is inside the
        goalmouth trapezoid defined by 4 image-space vertices [P1, P2, P3, P4].
        """
        if not ball_pixel or not polygon or len(polygon) < 3:
            return True  # Permissive fallback if polygon not specified
        
        u, v = ball_pixel
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if (v > min(p1y, p2y)) and (v <= max(p1y, p2y)):
                if p2y != p1y:
                    x_intersect = (v - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if u < x_intersect:
                        inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def _check_goal_hybrid(
        self,
        ball_pixel: Optional[Tuple[float, float]],
        ground_pos: Tuple[float, float],
        goal_side: str,  # 'left' or 'right'
        goal_polygon: Optional[List[List[float]]] = None,
        ball_velocity: Optional[Tuple[float, float]] = None
    ) -> Tuple[bool, float, str]:
        """
        Refined 2.5D Hybrid Geometric & Physics Check:
          - Condition 1 (Ground Footprint): (X_g, Y_g) crosses physical goal line within goal width.
          - Condition 2 (Image Pixel Polygon): (u, v) is inside 2D goalmouth polygon (under crossbar & inside posts).
          - Condition 3 (Velocity Vector): Ball moving toward attacking goal or resting in net.
        """
        if ground_pos is None:
            return False, 0.0, ""

        X_g, Y_g = ground_pos
        is_within_width = (self.goal_mouth_y_min <= Y_g <= self.goal_mouth_y_max)
        
        if goal_side == 'right':
            is_past_line = (X_g >= self.pitch_length - self.goal_line_thresh)
        else:
            is_past_line = (X_g <= self.goal_line_thresh)
            
        cond1_ground = is_past_line and is_within_width
        
        # Condition 2: Pixel-Space Point-In-Polygon Check
        cond2_pixel = False
        if ball_pixel is not None and goal_polygon is not None:
            cond2_pixel = self.point_in_goal_polygon(ball_pixel, goal_polygon)

        # Condition 3: Velocity Direction Check (if velocity is present)
        vel_ok = True
        if ball_velocity is not None:
            bvx, bvy = ball_velocity
            if goal_side == 'right' and bvx < -5.0:  # Moving strongly away from right goal
                vel_ok = False
            elif goal_side == 'left' and bvx > 5.0:   # Moving strongly away from left goal
                vel_ok = False

        # Trigger Goal if:
        # 1. Ground footprint confirms crossing within width (Condition 1 & 2), OR
        # 2. 2D visual polygon confirms ball is inside the net when in attacking third
        if vel_ok and ((cond1_ground and cond2_pixel) or (cond2_pixel and (X_g > self.pitch_length * 0.7 or X_g < self.pitch_length * 0.3)) or (cond1_ground and ball_pixel is None)):
            conf = 0.95 if ball_pixel is not None else 0.80
            side_str = "right" if goal_side == "right" else "left"
            return True, conf, f"2.5D Hybrid Goal ({side_str}): Ground ({X_g:.1f}m, {Y_g:.1f}m)"
            
        return False, 0.0, ""

