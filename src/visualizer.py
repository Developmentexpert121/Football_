"""
Stage 10 (Enhanced): Advanced Visualization with Pose Skeleton, Action Labels & Jersey Badges.

Renders overlay annotations onto video frames:
- Bounding boxes, track IDs, team color badges, speed text (km/h)
- Pose skeleton overlay (17 COCO keypoints + connections)
- Action labels (Running, Kicking, Sprinting, etc.)
- Jersey number badges (from OCR)
- Tactical mini-map pitch radar in screen corner
- Heatmap PNG rendering (per-team support)
- Enhanced event markers
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional


# Skeleton drawing configuration (matches pose_estimator.py)
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

# Edge colors (BGR) for different body parts
SKELETON_COLORS = {
    'face': (200, 200, 200),      # Light gray
    'arm_left': (50, 200, 50),    # Green
    'arm_right': (50, 50, 200),   # Red
    'torso': (200, 200, 50),      # Cyan
    'leg_left': (200, 100, 50),   # Blue
    'leg_right': (50, 100, 200),  # Orange
}

# Action label colors (BGR)
ACTION_COLORS = {
    'Standing': (180, 180, 180),
    'Walking': (200, 200, 100),
    'Running': (50, 200, 50),
    'Sprinting': (0, 255, 255),
    'Kicking': (0, 0, 255),
    'Sliding Tackle': (0, 100, 255),
    'Header': (255, 100, 0),
    'Dribbling': (255, 0, 255),
    'Unknown': (128, 128, 128),
}


class Visualizer:
    """
    Enhanced visualization engine with pose skeleton, action labels, and jersey badges.
    """
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        mini_map_w: int = 280,
        mini_map_h: int = 180,
        draw_pose: bool = True,
        draw_actions: bool = True,
        draw_jersey: bool = True
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.mini_map_w = mini_map_w
        self.mini_map_h = mini_map_h
        self.draw_pose = draw_pose
        self.draw_actions = draw_actions
        self.draw_jersey = draw_jersey
        
        self.color_team_a = (50, 50, 255)   # Red (BGR)
        self.color_team_b = (255, 100, 50)  # Blue (BGR)
        self.color_referee = (0, 255, 255)  # Yellow (BGR)
        self.color_ball = (0, 255, 0)       # Cyan/Green (BGR)

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        metric_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, int],
        player_stats: Dict[int, Dict[str, Any]],
        draw_mini_map: bool = True,
        camera_movement: Tuple[float, float] = (0.0, 0.0),
        possession_stats: Optional[Dict[str, Any]] = None,
        pose_data: Optional[Dict[int, Dict[str, Any]]] = None,
        action_labels: Optional[Dict[int, str]] = None,
        jersey_map: Optional[Dict[int, int]] = None,
        events_this_frame: Optional[List[Dict[str, Any]]] = None,
        smoothed_ball: Optional[Dict[str, Any]] = None,
        team_assigner: Optional[Any] = None,
        match_score: Tuple[int, int] = (0, 0),
        team_names: Optional[Tuple[str, str]] = None
    ) -> np.ndarray:
        """
        Draws bounding boxes, IDs, speeds, pose skeletons, action labels,
        jersey badges, broadcast scoreboard, and mini-map onto a single video frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Dynamic Team Colors and Names
        if team_assigner is not None:
            color_team_a = team_assigner.get_team_color(0)
            color_team_b = team_assigner.get_team_color(1)
            name_team_a = team_names[0] if team_names else team_assigner.get_team_name(0)
            name_team_b = team_names[1] if team_names else team_assigner.get_team_name(1)
        else:
            color_team_a = self.color_team_a
            color_team_b = self.color_team_b
            name_team_a = team_names[0] if team_names else "Team Red"
            name_team_b = team_names[1] if team_names else "Team White"

        # 0. Find ball and player in possession
        ball_pos_m = None
        for track in tracks:
            if track['class_id'] == 3 and track['track_id'] in metric_positions:
                ball_pos_m = metric_positions[track['track_id']]
                break
                
        player_in_possession_id = None
        if ball_pos_m is not None:
            min_dist = float('inf')
            for track in tracks:
                t_id = track['track_id']
                if track['class_id'] == 0 and t_id in metric_positions:
                    p_m = metric_positions[t_id]
                    dist = np.sqrt((ball_pos_m[0] - p_m[0])**2 + (ball_pos_m[1] - p_m[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        player_in_possession_id = t_id
            if min_dist > 2.0:
                player_in_possession_id = None

        # 1. Draw Pose Skeletons
        if self.draw_pose and pose_data:
            self._draw_pose_skeletons(annotated, pose_data, team_assignments)

        # 2. Draw Smoothed Ball indicator
        if smoothed_ball is not None and smoothed_ball.get('interpolated', False):
            bcx = int(smoothed_ball['cx'])
            bcy = int(smoothed_ball['cy'])
            cv2.circle(annotated, (bcx, bcy), 12, (0, 200, 255), 2)
            cv2.putText(annotated, "KF", (bcx - 8, bcy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

        # 3. Draw Player / Ball Overlays
        for track in tracks:
            bbox = track['bbox']
            t_id = track['track_id']
            cls_id = track['class_id']
            x1, y1, x2, y2 = map(int, bbox)

            if cls_id == 3:  # Ball
                bx, by = int((x1 + x2)/2), int(y1 - 10)
                pts = np.array([[bx-10, by-20], [bx+10, by-20], [bx, by]], np.int32)
                cv2.fillPoly(annotated, [pts], (0, 255, 0))
                cv2.polylines(annotated, [pts], True, (0, 0, 0), 2)
            else:
                # Color code team dynamically
                if cls_id == 2:  # Referee
                    color = self.color_referee
                    label_id = f"REF #{t_id}"
                    label_speed = ""
                    label_dist = ""
                elif cls_id == 1:  # Goalkeeper
                    # Determine Goalkeeper's team based on pitch half or team assignment
                    if t_id in metric_positions:
                        gk_x = metric_positions[t_id][0]
                        gk_team = 0 if gk_x <= self.pitch_length / 2 else 1
                    else:
                        gk_team = team_assignments.get(t_id, 0)
                    
                    color = (0, 215, 255)  # Gold/Yellow BGR for Goalkeeper ring
                    label_id = f"GK #{t_id}"
                    speed_km_h = player_stats.get(t_id, {}).get('avg_speed_km_h', 0.0) if t_id in player_stats else 0.0
                    dist_m = player_stats.get(t_id, {}).get('total_distance_m', 0.0) if t_id in player_stats else 0.0
                    label_speed = f"{speed_km_h} km/h" if speed_km_h > 0 else ""
                    label_dist = f"{dist_m} m" if dist_m > 0 else ""
                else:  # Player
                    team_id = team_assignments.get(t_id, 0)
                    color = color_team_a if team_id == 0 else color_team_b
                    
                    speed_km_h = 0.0
                    dist_m = 0.0
                    sprint_count = 0
                    if t_id in player_stats:
                        speed_km_h = player_stats[t_id].get('avg_speed_km_h', 0.0)
                        dist_m = player_stats[t_id].get('total_distance_m', 0.0)
                        sprint_count = player_stats[t_id].get('sprint_count', 0)
                    
                    label_id = f"{t_id}"
                    label_speed = f"{speed_km_h} km/h"
                    label_dist = f"{dist_m} m"

                # Draw ellipse indicator under feet
                foot_x = int((x1 + x2) / 2)
                foot_y = y2
                ring_thickness = 3 if cls_id == 1 else 2
                cv2.ellipse(annotated, (foot_x, foot_y), (max(12, int((x2 - x1)/2)), 7), 0, 0, 360, color, ring_thickness)

                # ----- Goalkeeper / Jersey Badge -----
                if cls_id == 1:
                    # Prominent GK badge above goalkeeper's head
                    gk_badge_text = f"GK #{t_id}"
                    badge_y_top = max(0, y1 - 32)
                    text_sz, _ = cv2.getTextSize(gk_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    badge_w = text_sz[0] + 16
                    badge_x = int((x1 + x2) / 2) - badge_w // 2
                    
                    # Gold background with dark text
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 24), (0, 215, 255), -1)
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 24), (0, 0, 0), 2)
                    cv2.putText(annotated, gk_badge_text, (badge_x + 8, badge_y_top + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                elif self.draw_jersey and jersey_map and t_id in jersey_map:
                    jersey_num = str(jersey_map[t_id])
                    badge_y_top = max(0, y1 - 30)
                    badge_w = len(jersey_num) * 14 + 10
                    badge_x = int((x1 + x2) / 2) - badge_w // 2
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), color, -1)
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (0, 0, 0), 1)
                    cv2.putText(annotated, f"#{jersey_num}", (badge_x + 3, badge_y_top + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # ----- Track ID Badge below feet -----
                badge_w = len(label_id) * 12 + 10
                cv2.rectangle(annotated, (foot_x - badge_w//2, foot_y + 5), (foot_x + badge_w//2, foot_y + 25), (255, 255, 255), -1)
                cv2.rectangle(annotated, (foot_x - badge_w//2, foot_y + 5), (foot_x + badge_w//2, foot_y + 25), color, 1)
                cv2.putText(annotated, label_id, (foot_x - badge_w//2 + 5, foot_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw speed and distance under the badge
                if label_speed:
                    text_size, _ = cv2.getTextSize(label_speed, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                    cv2.putText(annotated, label_speed, (foot_x - text_size[0]//2, foot_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                if label_dist:
                    text_size, _ = cv2.getTextSize(label_dist, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
                    cv2.putText(annotated, label_dist, (foot_x - text_size[0]//2, foot_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)

                # ----- Action Label above player -----
                if self.draw_actions and action_labels and t_id in action_labels:
                    action = action_labels[t_id]
                    if action not in ('Standing', 'Unknown'):  # Skip boring labels
                        action_color = ACTION_COLORS.get(action, (128, 128, 128))
                        action_y = max(0, y1 - 8)
                        if self.draw_jersey and jersey_map and t_id in jersey_map:
                            action_y = max(0, y1 - 38)  # Move above jersey badge
                        text_size, _ = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        action_x = int((x1 + x2) / 2) - text_size[0] // 2

                        # Semi-transparent background
                        overlay = annotated.copy()
                        cv2.rectangle(overlay, (action_x - 2, action_y - 14),
                                      (action_x + text_size[0] + 2, action_y + 2), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

                        cv2.putText(annotated, action, (action_x, action_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_color, 2)

                # If player is in possession, draw red triangle over head
                if t_id == player_in_possession_id:
                    hx, hy = int((x1 + x2)/2), int(y1 - 10)
                    if self.draw_jersey and jersey_map and t_id in jersey_map:
                        hy = max(0, y1 - 38)
                    if self.draw_actions and action_labels and t_id in action_labels:
                        hy -= 18
                    pts = np.array([[hx-10, hy-20], [hx+10, hy-20], [hx, hy]], np.int32)
                    cv2.fillPoly(annotated, [pts], (0, 0, 255))  # Red
                    cv2.polylines(annotated, [pts], True, (0, 0, 0), 2)

        # 4. Draw Top-Down Tactical Pitch Mini-Map
        if draw_mini_map:
            mini_map = self._render_mini_map(metric_positions, team_assignments, player_in_possession_id, ball_pos_m)
            # Overlay in top-right corner
            margin = 15
            y_end = min(margin + self.mini_map_h, h)
            x_start = max(w - margin - self.mini_map_w, 0)
            mh = y_end - margin
            mw = w - margin - x_start
            if mh > 0 and mw > 0:
                annotated[margin:y_end, x_start:w - margin] = mini_map[:mh, :mw]

        # 5. Draw Top Broadcast Scoreboard Header (Live Goals & Team Names)
        self._draw_scoreboard(annotated, match_score, (name_team_a, name_team_b), (color_team_a, color_team_b), h, w)

        # 6. Draw Camera Movement and Possession stats (HUD)
        self._draw_hud(annotated, camera_movement, possession_stats, h, w, team_names=(name_team_a, name_team_b))

        # 7. Draw event markers on frame
        if events_this_frame:
            self._draw_event_markers(annotated, events_this_frame, h, w)

        return annotated

    def _draw_scoreboard(
        self,
        frame: np.ndarray,
        score: Tuple[int, int],
        team_names: Tuple[str, str],
        team_colors: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        h: int, w: int
    ):
        """Draws top broadcast scoreboard header displaying live match score & team names."""
        team_a_name, team_b_name = team_names
        score_a, score_b = score
        color_a, color_b = team_colors

        board_w = 480
        board_h = 44
        center_x = w // 2
        start_x = center_x - board_w // 2
        start_y = 12

        # Dark glassmorphism background banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + board_w, start_y + board_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (start_x, start_y), (start_x + board_w, start_y + board_h), (220, 220, 220), 1)

        # Team A Badge Box (Left)
        cv2.rectangle(frame, (start_x + 6, start_y + 6), (start_x + 165, start_y + 38), color_a, -1)
        cv2.putText(frame, team_a_name[:12].upper(), (start_x + 12, start_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Score A Box
        cv2.rectangle(frame, (start_x + 170, start_y + 6), (start_x + 215, start_y + 38), (255, 255, 255), -1)
        cv2.putText(frame, str(score_a), (start_x + 185, start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # Separator ":"
        cv2.putText(frame, ":", (center_x - 4, start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Score B Box
        cv2.rectangle(frame, (start_x + 265, start_y + 6), (start_x + 310, start_y + 38), (255, 255, 255), -1)
        cv2.putText(frame, str(score_b), (start_x + 280, start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

        # Team B Badge Box (Right)
        cv2.rectangle(frame, (start_x + 315, start_y + 6), (start_x + 474, start_y + 38), color_b, -1)
        cv2.putText(frame, team_b_name[:12].upper(), (start_x + 321, start_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def _draw_pose_skeletons(
        self,
        frame: np.ndarray,
        pose_data: Dict[int, Dict[str, Any]],
        team_assignments: Dict[int, int]
    ):
        """
        Draws pose skeleton overlays for all tracked players.
        """
        for t_id, pdata in pose_data.items():
            kps = pdata.get('keypoints')
            if kps is None:
                continue

            team_id = team_assignments.get(t_id, 0)
            base_color = self.color_team_a if team_id == 0 else self.color_team_b

            # Draw skeleton edges
            for i, (start_idx, end_idx) in enumerate(SKELETON_EDGES):
                if kps[start_idx, 2] < 0.3 or kps[end_idx, 2] < 0.3:
                    continue

                pt1 = (int(kps[start_idx, 0]), int(kps[start_idx, 1]))
                pt2 = (int(kps[end_idx, 0]), int(kps[end_idx, 1]))

                # Color by body part
                if i < 4:
                    edge_color = SKELETON_COLORS['face']
                elif i == 4:
                    edge_color = SKELETON_COLORS['torso']
                elif i < 7:
                    edge_color = SKELETON_COLORS['arm_left']
                elif i < 9:
                    edge_color = SKELETON_COLORS['arm_right']
                elif i < 12:
                    edge_color = SKELETON_COLORS['torso']
                elif i < 14:
                    edge_color = SKELETON_COLORS['leg_left']
                else:
                    edge_color = SKELETON_COLORS['leg_right']

                cv2.line(frame, pt1, pt2, edge_color, 2, cv2.LINE_AA)

            # Draw keypoint dots
            for j in range(17):
                if kps[j, 2] < 0.3:
                    continue
                pt = (int(kps[j, 0]), int(kps[j, 1]))
                cv2.circle(frame, pt, 3, base_color, -1)
                cv2.circle(frame, pt, 3, (0, 0, 0), 1)

    def _draw_hud(
        self,
        frame: np.ndarray,
        camera_movement: Tuple[float, float],
        possession_stats: Optional[Dict[str, Any]],
        h: int, w: int,
        team_names: Optional[Tuple[str, str]] = None
    ):
        """Draws heads-up display with camera movement and possession stats."""
        # Camera Movement (top-left)
        cm_text_x = f"Camera Movement X: {camera_movement[0]:.2f}"
        cm_text_y = f"Camera Movement Y: {camera_movement[1]:.2f}"
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, cm_text_x, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, cm_text_y, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Possession Stats (bottom-right)
        if possession_stats:
            team_a_name = team_names[0] if team_names else "Team 1"
            team_b_name = team_names[1] if team_names else "Team 2"
            team_a_pct = possession_stats.get('team_a_possession_pct', 0.0)
            team_b_pct = possession_stats.get('team_b_possession_pct', 0.0)
            p_text_a = f"{team_a_name} Possession: {team_a_pct:.1f}%"
            p_text_b = f"{team_b_name} Possession: {team_b_pct:.1f}%"
            
            overlay2 = frame.copy()
            box_h = 80
            box_w = 480
            start_y = h - box_h - 20
            start_x = w - box_w - 20
            cv2.rectangle(overlay2, (start_x, start_y), (start_x + box_w, start_y + box_h), (200, 200, 200), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, p_text_a, (start_x + 15, start_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, p_text_b, (start_x + 15, start_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def _draw_event_markers(
        self,
        frame: np.ndarray,
        events: List[Dict[str, Any]],
        h: int, w: int
    ):
        """Draws event notification banners on the frame."""
        event_color_map = {
            'Goal': (0, 255, 0),
            'Shot on Target': (0, 200, 255),
            'Potential Foul': (0, 165, 255),
            'Yellow Card Candidate': (0, 255, 255),
            'Corner Kick': (255, 200, 0),
            'Free Kick': (200, 200, 50),
            'Offside': (255, 0, 0),
            'Penalty Area Entry': (255, 100, 100),
        }

        for idx, event in enumerate(events[:3]):  # Max 3 events per frame
            event_type = event.get('event_type', '')
            color = event_color_map.get(event_type, (200, 200, 200))
            y_offset = 90 + idx * 45

            # Semi-transparent banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y_offset), (500, y_offset + 38), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Event type badge
            cv2.rectangle(frame, (12, y_offset + 2), (12 + len(event_type) * 10 + 10, y_offset + 22), color, -1)
            cv2.putText(frame, event_type, (17, y_offset + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Confidence
            conf = event.get('confidence', 0)
            cv2.putText(frame, f"{int(conf * 100)}%",
                        (12 + len(event_type) * 10 + 15, y_offset + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Description
            desc = event.get('description', '')[:60]
            cv2.putText(frame, desc, (17, y_offset + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    def _render_mini_map(
        self,
        metric_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, int],
        possession_id: Optional[int] = None,
        ball_pos_m: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Renders 2D tactical mini-map diagram showing pitch coordinates.
        Enhanced with ball position and possession indicator.
        """
        mini_map = np.full((self.mini_map_h, self.mini_map_w, 3), (34, 139, 34), dtype=np.uint8)

        # Pitch outline
        cv2.rectangle(mini_map, (5, 5), (self.mini_map_w - 5, self.mini_map_h - 5), (255, 255, 255), 2)
        # Center line
        cv2.line(mini_map, (self.mini_map_w // 2, 5), (self.mini_map_w // 2, self.mini_map_h - 5), (255, 255, 255), 1)
        # Center circle
        cv2.circle(mini_map, (self.mini_map_w // 2, self.mini_map_h // 2), 20, (255, 255, 255), 1)

        # Penalty areas (scaled proportions)
        pa_w = int((16.5 / self.pitch_length) * (self.mini_map_w - 10))
        pa_h = int((40.32 / self.pitch_width) * (self.mini_map_h - 10))
        pa_y_offset = (self.mini_map_h - pa_h) // 2
        # Left penalty area
        cv2.rectangle(mini_map, (5, pa_y_offset), (5 + pa_w, pa_y_offset + pa_h), (255, 255, 255), 1)
        # Right penalty area
        cv2.rectangle(mini_map, (self.mini_map_w - 5 - pa_w, pa_y_offset), (self.mini_map_w - 5, pa_y_offset + pa_h), (255, 255, 255), 1)

        # Draw ball
        if ball_pos_m is not None:
            bpx = int(5 + (ball_pos_m[0] / self.pitch_length) * (self.mini_map_w - 10))
            bpy = int(5 + (ball_pos_m[1] / self.pitch_width) * (self.mini_map_h - 10))
            bpx = max(5, min(self.mini_map_w - 5, bpx))
            bpy = max(5, min(self.mini_map_h - 5, bpy))
            cv2.circle(mini_map, (bpx, bpy), 4, (0, 255, 255), -1)
            cv2.circle(mini_map, (bpx, bpy), 4, (0, 0, 0), 1)

        # Draw players
        for t_id, (x_m, y_m) in metric_positions.items():
            px = int(5 + (x_m / self.pitch_length) * (self.mini_map_w - 10))
            py = int(5 + (y_m / self.pitch_width) * (self.mini_map_h - 10))
            px = max(5, min(self.mini_map_w - 5, px))
            py = max(5, min(self.mini_map_h - 5, py))

            team_id = team_assignments.get(t_id, 0)
            color = self.color_team_a if team_id == 0 else self.color_team_b

            radius = 5
            if t_id == possession_id:
                radius = 7
                cv2.circle(mini_map, (px, py), radius + 2, (0, 0, 255), 2)  # Red ring

            cv2.circle(mini_map, (px, py), radius, color, -1)
            cv2.circle(mini_map, (px, py), radius, (0, 0, 0), 1)

        return mini_map

    @staticmethod
    def generate_heatmap_image(
        player_stats: Dict[int, Dict[str, Any]],
        output_path: str = "reports/heatmap.png",
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        team_id: Optional[int] = None,
        title_suffix: str = ""
    ):
        """
        Generates 2D Matplotlib density heatmap PNG saved to reports directory.
        Supports per-team filtering.
        """
        all_x, all_y = [], []
        for p_id, stats in player_stats.items():
            if team_id is not None and stats.get('team_id') != team_id:
                continue
            positions = stats.get('positions', [])
            for x, y in positions:
                all_x.append(x)
                all_y.append(y)

        if not all_x:
            all_x = [52.5]
            all_y = [34.0]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.set_facecolor('#2e8b57')
        
        # Draw 2D histogram heatmap
        counts, xedges, yedges, im = ax.hist2d(
            all_x, all_y, bins=[35, 22],
            range=[[0, pitch_length], [0, pitch_width]],
            cmap='YlOrRd', alpha=0.85
        )

        # Draw pitch markings on heatmap
        ax.axvline(x=pitch_length / 2, color='white', linewidth=0.5, alpha=0.5)
        circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, fill=False, color='white', linewidth=0.5, alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        title = f"Football Match Position Heatmap{title_suffix}"
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel("Pitch Length (meters)", color='white')
        ax.set_ylabel("Pitch Width (meters)", color='white')
        fig.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')

        plt.colorbar(im, ax=ax, label="Position Density")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"Position heatmap PNG saved to: {output_path}")
