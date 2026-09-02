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
        draw_jersey: bool = True,
        draw_goal_overlay: bool = True,
        draw_debug_coordinates: bool = False,
        left_goal_polygon: Optional[List[List[float]]] = None,
        right_goal_polygon: Optional[List[List[float]]] = None,
        left_net_roi: Optional[List[int]] = None,
        right_net_roi: Optional[List[int]] = None,
        reference_points_image: Optional[List[List[float]]] = None,
        reference_points_pitch: Optional[List[List[float]]] = None
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.mini_map_w = mini_map_w
        self.mini_map_h = mini_map_h
        self.draw_pose = draw_pose
        self.draw_actions = draw_actions
        self.draw_jersey = draw_jersey
        self.draw_goal_overlay = False  # Disabled static goal overlay to prevent screen clutter during camera angle changes
        self.draw_debug_coordinates = draw_debug_coordinates

        # Net ROI pixel coordinates [x1, y1, x2, y2]
        self.left_net_roi = left_net_roi or [50, 210, 160, 420]
        self.right_net_roi = right_net_roi or [1120, 210, 1230, 420]
        # Homography reference points (image pixels)
        self.reference_points_image = reference_points_image

        # Goal mouth metric dimensions (FIFA standard)
        self.goal_width_m = 7.32       # 7.32m wide
        self.goal_height_m = 2.44      # 2.44m crossbar height
        self.goal_depth_m = 2.44       # net depth behind goal line
        self.goal_y_min_m = (pitch_width - self.goal_width_m) / 2.0   # 30.34m
        self.goal_y_max_m = (pitch_width + self.goal_width_m) / 2.0   # 37.66m

        # Goal flash animation state
        self._goal_flash_counter = 0
        self._goal_flash_side = None

        self.color_team_a = (50, 50, 255)   # Red (BGR)
        self.color_team_b = (255, 100, 50)  # Blue (BGR)
        self.color_referee = (0, 255, 255)  # Yellow (BGR)
        self.color_ball = (0, 255, 0)       # Cyan/Green (BGR)

        # ── Compute inverse homography for 3D goal projection ──
        self.H_inv = None
        self.left_goal_3d = None   # dict of 3D goal pixel coords
        self.right_goal_3d = None
        ref_img = reference_points_image
        ref_pitch = reference_points_pitch
        if ref_img and ref_pitch and len(ref_img) >= 4 and len(ref_pitch) >= 4:
            try:
                H, _ = cv2.findHomography(
                    np.float32(ref_img), np.float32(ref_pitch)
                )
                self.H_inv = np.linalg.inv(H)
                self.left_goal_3d = self._compute_3d_goal_pixels('left')
                self.right_goal_3d = self._compute_3d_goal_pixels('right')
            except Exception as e:
                print(f"[Visualizer] Could not compute 3D goal projection: {e}")

        # Fallback to config polygons if 3D projection failed
        self.left_goal_polygon = left_goal_polygon or [
            [97, 392], [150, 392], [150, 230], [97, 230]
        ]
        self.right_goal_polygon = right_goal_polygon or [
            [1129, 392], [1182, 392], [1182, 230], [1129, 230]
        ]

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
        team_names: Optional[Tuple[str, str]] = None,
        ball_metric_pos: Optional[Tuple[float, float]] = None
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
            self._draw_pose_skeletons(annotated, pose_data, team_assignments, tracks)

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
                    if t_id in player_stats:
                        speed_km_h = player_stats[t_id].get('avg_speed_km_h', 0.0)
                        dist_m = player_stats[t_id].get('total_distance_m', 0.0)
                    
                    # Primary label uses OCR Jersey Number if available, fallback to Track ID
                    if self.draw_jersey and jersey_map and t_id in jersey_map:
                        label_id = f"#{jersey_map[t_id]}"
                    else:
                        label_id = f"#{t_id}"

                    label_speed = f"{speed_km_h} km/h" if speed_km_h > 0 else ""
                    label_dist = f"{dist_m} m" if dist_m > 0 else ""

                # Draw ellipse indicator under feet
                foot_x = int((x1 + x2) / 2)
                foot_y = y2
                ring_thickness = 3 if cls_id in (1, 2) else 2
                cv2.ellipse(annotated, (foot_x, foot_y), (max(12, int((x2 - x1)/2)), 7), 0, 0, 360, color, ring_thickness)

                # ----- Primary Player / GK / Ref Head Badge -----
                badge_y_top = max(5, y1 - 28)
                if cls_id == 1:
                    # Prominent GK badge
                    gk_j = jersey_map.get(t_id, t_id) if (jersey_map and t_id in jersey_map) else t_id
                    gk_badge_text = f"GK #{gk_j}"
                    text_sz, _ = cv2.getTextSize(gk_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    badge_w = text_sz[0] + 14
                    badge_x = int((x1 + x2) / 2) - badge_w // 2
                    
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (0, 215, 255), -1)
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (0, 0, 0), 2)
                    cv2.putText(annotated, gk_badge_text, (badge_x + 7, badge_y_top + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                elif cls_id == 2:
                    # Prominent Referee badge
                    ref_badge_text = f"REF #{t_id}"
                    text_sz, _ = cv2.getTextSize(ref_badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    badge_w = text_sz[0] + 14
                    badge_x = int((x1 + x2) / 2) - badge_w // 2
                    
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (0, 255, 255), -1)
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (0, 0, 0), 2)
                    cv2.putText(annotated, ref_badge_text, (badge_x + 7, badge_y_top + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                else:
                    # Player Jersey Number Badge (White box with bold team-colored text)
                    badge_text = label_id
                    text_sz, _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    badge_w = max(28, text_sz[0] + 12)
                    badge_x = int((x1 + x2) / 2) - badge_w // 2

                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), (255, 255, 255), -1)
                    cv2.rectangle(annotated, (badge_x, badge_y_top), (badge_x + badge_w, badge_y_top + 22), color, 2)
                    cv2.putText(annotated, badge_text, (badge_x + 6, badge_y_top + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # ----- Speed and Distance Metrics below feet -----
                if label_speed:
                    text_size, _ = cv2.getTextSize(label_speed, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
                    cv2.putText(annotated, label_speed, (foot_x - text_size[0]//2, foot_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)
                if label_dist:
                    text_size, _ = cv2.getTextSize(label_dist, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
                    cv2.putText(annotated, label_dist, (foot_x - text_size[0]//2, foot_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 2)

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

        # 8. Draw 3D Goal Polygon Overlay on video frame
        if self.draw_goal_overlay:
            has_goal_event = False
            if events_this_frame:
                has_goal_event = any(e.get('event_type') == 'Goal' for e in events_this_frame)

            # Trigger goal flash animation
            if has_goal_event:
                self._goal_flash_counter = 30  # flash for 30 frames
                goal_evt = next((e for e in events_this_frame if e.get('event_type') == 'Goal'), None)
                self._goal_flash_side = goal_evt.get('goal_side', 'left') if goal_evt else 'left'

            # Draw goal polygons when camera/ball is in penalty zone
            ball_near_left = True
            ball_near_right = True
            if ball_metric_pos is not None:
                bx = ball_metric_pos[0]
                ball_near_left = bx < 35.0
                ball_near_right = bx > (self.pitch_length - 35.0)

            if ball_near_left or self._goal_flash_counter > 0:
                self._draw_goal_polygon_on_frame(
                    annotated, self.left_goal_polygon, 'LEFT GOAL',
                    is_goal_flash=(self._goal_flash_counter > 0 and self._goal_flash_side == 'left'),
                    goal_3d=self.left_goal_3d
                )
            if ball_near_right or self._goal_flash_counter > 0:
                self._draw_goal_polygon_on_frame(
                    annotated, self.right_goal_polygon, 'RIGHT GOAL',
                    is_goal_flash=(self._goal_flash_counter > 0 and self._goal_flash_side == 'right'),
                    goal_3d=self.right_goal_3d
                )

            if self._goal_flash_counter > 0:
                self._goal_flash_counter -= 1
                # Draw big "GOAL!" flash banner (center screen)
                if self._goal_flash_counter > 15:
                    overlay_flash = annotated.copy()
                    cv2.rectangle(overlay_flash, (w // 2 - 160, h // 2 - 40), (w // 2 + 160, h // 2 + 40), (0, 255, 0), -1)
                    cv2.addWeighted(overlay_flash, 0.4, annotated, 0.6, 0, annotated)
                    cv2.putText(annotated, "GOAL!", (w // 2 - 100, h // 2 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)

        # 9. Draw debug coordinate overlay (reference points, goal vertices, net ROIs)
        if self.draw_debug_coordinates:
            self._draw_coordinate_debug_overlay(annotated)

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
        team_assignments: Dict[int, int],
        tracks: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Draws pose skeleton overlays specifically for Goalkeepers.
        """
        gk_track_ids = set()
        if tracks:
            gk_track_ids = {t['track_id'] for t in tracks if t.get('class_id') == 1}

        for t_id, pdata in pose_data.items():
            if gk_track_ids and t_id not in gk_track_ids:
                continue

            kps = pdata.get('keypoints')
            if kps is None:
                continue

            base_color = (0, 215, 255)  # Gold highlight for Goalkeeper skeleton

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

    def _pitch_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Project pitch meters (ground plane) to image pixels via H_inv."""
        pt = np.float32([[[x_m, y_m]]])
        px = cv2.perspectiveTransform(pt, self.H_inv)
        return int(px[0][0][0]), int(px[0][0][1])

    def _compute_3d_goal_pixels(self, side: str) -> dict:
        """
        Compute pixel coordinates for a full 3D goal wireframe.
        Uses inverse homography for ground plane + estimated vertical scale for crossbar.

        Returns dict with keys:
          post_left_base, post_right_base (ground level post feet)
          post_left_top, post_right_top (crossbar corners)
          net_back_left, net_back_right (back of net, ground level)
          net_back_left_top, net_back_right_top (back of net, crossbar height)
          crossbar_height_px (estimated crossbar pixel offset)
        """
        if self.H_inv is None:
            return None

        y_min = self.goal_y_min_m  # 30.34
        y_max = self.goal_y_max_m  # 37.66
        y_mid = (y_min + y_max) / 2.0

        if side == 'left':
            goal_x = 0.0
            net_x = -self.goal_depth_m  # behind the goal line
        else:
            goal_x = self.pitch_length   # 105.0
            net_x = self.pitch_length + self.goal_depth_m

        # Ground-level points
        post_left_base = self._pitch_to_pixel(goal_x, y_min)
        post_right_base = self._pitch_to_pixel(goal_x, y_max)
        net_back_left = self._pitch_to_pixel(net_x, y_min)
        net_back_right = self._pitch_to_pixel(net_x, y_max)

        # Estimate crossbar height in pixels using local vertical scale
        # Take two points 1m apart on the y-axis at goal line position
        p_a = self._pitch_to_pixel(goal_x, y_mid)
        p_b = self._pitch_to_pixel(goal_x, y_mid + 1.0)
        pixels_per_meter = np.sqrt((p_b[0] - p_a[0])**2 + (p_b[1] - p_a[1])**2)
        # Use 1.2x multiplier and minimum 45px so goal is clearly visible
        crossbar_px = max(45, int(self.goal_height_m * pixels_per_meter * 1.2))

        # Crossbar pixel positions (shift upward from base)
        post_left_top = (post_left_base[0], post_left_base[1] - crossbar_px)
        post_right_top = (post_right_base[0], post_right_base[1] - crossbar_px)

        # Back of net top (70% height for perspective diminishing effect)
        back_crossbar_px = int(crossbar_px * 0.75)
        net_back_left_top = (net_back_left[0], net_back_left[1] - back_crossbar_px)
        net_back_right_top = (net_back_right[0], net_back_right[1] - back_crossbar_px)

        return {
            'post_left_base': post_left_base,
            'post_right_base': post_right_base,
            'post_left_top': post_left_top,
            'post_right_top': post_right_top,
            'net_back_left': net_back_left,
            'net_back_right': net_back_right,
            'net_back_left_top': net_back_left_top,
            'net_back_right_top': net_back_right_top,
            'crossbar_height_px': crossbar_px,
        }

    def _draw_goal_polygon_on_frame(
        self,
        frame: np.ndarray,
        polygon: List[List[float]],
        label: str,
        is_goal_flash: bool = False,
        goal_3d: dict = None
    ):
        """
        Draws a 3D goal wireframe on the video frame if 3D data is available,
        otherwise falls back to the flat polygon overlay.
        Shows: posts, crossbar, net depth, side netting, with goal-scored flash.
        """
        if goal_3d is not None:
            self._draw_3d_goal_wireframe(frame, goal_3d, label, is_goal_flash)
            return

        # Disable static fallback boxes at screen edges
        return

    def _draw_3d_goal_wireframe(
        self,
        frame: np.ndarray,
        g: dict,
        label: str,
        is_goal_flash: bool = False
    ):
        """
        Draws a full 3D perspective goal wireframe:
        - Two vertical posts (white thick lines)
        - Horizontal crossbar (white thick line)
        - Net depth: side netting (4 lines connecting front to back)
        - Back of net rectangle
        - Semi-transparent goal mouth fill
        - NET mesh inside the back plane
        - GOAL! flash effect when goal is scored
        """
        # Unpack coordinates
        plb = g['post_left_base']
        prb = g['post_right_base']
        plt_ = g['post_left_top']
        prt = g['post_right_top']
        nbl = g['net_back_left']
        nbr = g['net_back_right']
        nblt = g['net_back_left_top']
        nbrt = g['net_back_right_top']

        # ── Colors ──
        if is_goal_flash:
            post_color = (0, 255, 0)
            net_color = (0, 200, 0)
            fill_color = (0, 255, 0)
            fill_alpha = 0.40
            label_color = (0, 255, 0)
        else:
            post_color = (255, 255, 255)
            net_color = (180, 180, 180)
            fill_color = (200, 255, 200)
            fill_alpha = 0.12
            label_color = (0, 255, 255)

        # ── 1. Semi-transparent goal mouth fill (front face) ──
        front_face = np.array([plb, prb, prt, plt_], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [front_face], fill_color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

        # ── 2. Back of net fill (darker, semi-transparent) ──
        back_face = np.array([nbl, nbr, nbrt, nblt], dtype=np.int32)
        overlay2 = frame.copy()
        cv2.fillPoly(overlay2, [back_face], (40, 40, 40))
        cv2.addWeighted(overlay2, 0.3, frame, 0.7, 0, frame)

        # ── 3. Side netting lines (connect front to back) ──
        side_lines = [
            (plb, nbl), (prb, nbr),       # bottom edges
            (plt_, nblt), (prt, nbrt),     # top edges
        ]
        for p1, p2 in side_lines:
            cv2.line(frame, p1, p2, net_color, 1, cv2.LINE_AA)

        # ── 4. Net mesh on back plane ──
        n_mesh = 5
        for i in range(1, n_mesh):
            frac = i / n_mesh
            # Horizontal mesh line
            lx = int(nbl[0] + (nbr[0] - nbl[0]) * frac)
            ly = int(nbl[1] + (nbr[1] - nbl[1]) * frac)
            ltx = int(nblt[0] + (nbrt[0] - nblt[0]) * frac)
            lty = int(nblt[1] + (nbrt[1] - nblt[1]) * frac)
            cv2.line(frame, (lx, ly), (ltx, lty), (100, 100, 100), 1)
        for i in range(1, n_mesh):
            frac = i / n_mesh
            # Vertical mesh line
            bx = int(nbl[0] + (nblt[0] - nbl[0]) * frac)
            by = int(nbl[1] + (nblt[1] - nbl[1]) * frac)
            brx = int(nbr[0] + (nbrt[0] - nbr[0]) * frac)
            bry = int(nbr[1] + (nbrt[1] - nbr[1]) * frac)
            cv2.line(frame, (bx, by), (brx, bry), (100, 100, 100), 1)

        # ── 5. Net mesh on top plane (roof netting) ──
        top_face = [plt_, prt, nbrt, nblt]
        for i in range(1, 4):
            frac = i / 4
            fx = int(plt_[0] + (nblt[0] - plt_[0]) * frac)
            fy = int(plt_[1] + (nblt[1] - plt_[1]) * frac)
            gx = int(prt[0] + (nbrt[0] - prt[0]) * frac)
            gy = int(prt[1] + (nbrt[1] - prt[1]) * frac)
            cv2.line(frame, (fx, fy), (gx, gy), (100, 100, 100), 1)

        # ── 6. Back of net border ──
        cv2.polylines(frame, [back_face], True, net_color, 1, cv2.LINE_AA)

        # ── 7. POSTS (thick white lines) ──
        cv2.line(frame, plb, plt_, post_color, 3, cv2.LINE_AA)
        cv2.line(frame, prb, prt, post_color, 3, cv2.LINE_AA)

        # ── 8. CROSSBAR (thick white line) ──
        cv2.line(frame, plt_, prt, post_color, 4, cv2.LINE_AA)

        # ── 9. Post base circles (ground contact) ──
        cv2.circle(frame, plb, 4, post_color, -1)
        cv2.circle(frame, prb, 4, post_color, -1)

        # ── 10. GOAL! flash big text on screen ──
        if is_goal_flash:
            h, w = frame.shape[:2]
            # Draw "GOAL!" on crossbar area
            bar_cx = (plt_[0] + prt[0]) // 2
            bar_cy = (plt_[1] + prt[1]) // 2
            cv2.putText(frame, "GOAL!", (bar_cx - 50, bar_cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    def _draw_coordinate_debug_overlay(self, frame: np.ndarray):
        """
        Draws debug markers for all calibrated coordinates:
        - Homography reference points (TL, TR, BR, BL)
        - Goal polygon vertices (P1-P4 for left and right)
        - Net ROI rectangles
        """
        h, w = frame.shape[:2]

        # 1. Homography reference points
        if self.reference_points_image:
            ref_labels = ['TL', 'TR', 'BR', 'BL']
            for i, pt in enumerate(self.reference_points_image[:4]):
                px, py = int(pt[0]), int(pt[1])
                cv2.drawMarker(frame, (px, py), (0, 200, 200), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(frame, f"REF-{ref_labels[i]}", (px + 10, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

        # 2. Left goal polygon
        if self.left_goal_polygon and len(self.left_goal_polygon) >= 4:
            pts = np.array(self.left_goal_polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 165, 255), 1, cv2.LINE_AA)
            for i, (px, py) in enumerate(self.left_goal_polygon[:4]):
                cv2.circle(frame, (int(px), int(py)), 3, (0, 165, 255), -1)
                cv2.putText(frame, f"LG-P{i+1}", (int(px) + 6, int(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)

        # 3. Right goal polygon
        if self.right_goal_polygon and len(self.right_goal_polygon) >= 4:
            pts = np.array(self.right_goal_polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 165, 0), 1, cv2.LINE_AA)
            for i, (px, py) in enumerate(self.right_goal_polygon[:4]):
                cv2.circle(frame, (int(px), int(py)), 3, (255, 165, 0), -1)
                cv2.putText(frame, f"RG-P{i+1}", (int(px) + 6, int(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 165, 0), 1)

        # 4. Net ROI rectangles
        if self.left_net_roi:
            x1, y1, x2, y2 = self.left_net_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 255, 128), 1)
            cv2.putText(frame, "L-NET-ROI", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
        if self.right_net_roi:
            x1, y1, x2, y2 = self.right_net_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 255), 1)
            cv2.putText(frame, "R-NET-ROI", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 255), 1)

    def _render_mini_map(
        self,
        metric_positions: Dict[int, Tuple[float, float]],
        team_assignments: Dict[int, int],
        possession_id: Optional[int] = None,
        ball_pos_m: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Renders 2D tactical mini-map diagram showing pitch coordinates.
        Enhanced with prominent 3D goal nets, ball position and possession indicator.
        """
        map_w = self.mini_map_w
        map_h = self.mini_map_h
        # Use extra padding so goal nets have room to draw OUTSIDE the pitch
        pad = 12  # padding from edge of image to pitch outline
        pitch_w_px = map_w - 2 * pad   # usable pitch pixel width
        pitch_h_px = map_h - 2 * pad   # usable pitch pixel height

        mini_map = np.full((map_h, map_w, 3), (20, 100, 20), dtype=np.uint8)

        # Helper: pitch meters -> mini-map pixels (within pitch area)
        def m2px(x_m, y_m):
            px = int(pad + (x_m / self.pitch_length) * pitch_w_px)
            py = int(pad + (y_m / self.pitch_width) * pitch_h_px)
            return max(0, min(map_w - 1, px)), max(0, min(map_h - 1, py))

        # ── GOAL DIMENSIONS in pixels ──────────────────────────
        # Goal mouth: 7.32m wide, centered at y = pitch_width/2
        gy_top = int(pad + (self.goal_y_min_m / self.pitch_width) * pitch_h_px)
        gy_bot = int(pad + (self.goal_y_max_m / self.pitch_width) * pitch_h_px)
        goal_h_px = gy_bot - gy_top  # ~17-20px depending on map size

        # Net depth: exaggerate for visibility (min 10px)
        net_depth = max(10, int(pad * 0.9))

        # ═══════════════════════════════════════════════════════
        # LEFT GOAL (x=0 side)
        # ═══════════════════════════════════════════════════════
        lg_line_x = pad  # goal line x position
        lg_net_x = lg_line_x - net_depth  # back of net

        # 3D trapezoid net (slightly narrower at back for depth effect)
        shrink = 3  # pixels narrower at the back
        trap_pts_l = np.array([
            [lg_line_x, gy_top],      # front-top (goal post)
            [lg_line_x, gy_bot],      # front-bottom (goal post)
            [lg_net_x, gy_bot - shrink],   # back-bottom
            [lg_net_x, gy_top + shrink],   # back-top
        ], dtype=np.int32)

        # Dark net background fill
        cv2.fillPoly(mini_map, [trap_pts_l], (50, 50, 50))
        # Net mesh: horizontal
        for ny in range(gy_top + shrink, gy_bot - shrink, 3):
            cv2.line(mini_map, (lg_net_x, ny), (lg_line_x, ny), (120, 120, 120), 1)
        # Net mesh: vertical
        for nx in range(lg_net_x, lg_line_x, 3):
            frac_x = (nx - lg_net_x) / max(1, (lg_line_x - lg_net_x))
            top_y = int(gy_top + shrink * (1 - frac_x))
            bot_y = int(gy_bot - shrink * (1 - frac_x))
            cv2.line(mini_map, (nx, top_y), (nx, bot_y), (120, 120, 120), 1)
        # Net border (trapezoid outline)
        cv2.polylines(mini_map, [trap_pts_l], True, (200, 200, 200), 1)
        # Goal posts (thick bright dots)
        cv2.circle(mini_map, (lg_line_x, gy_top), 3, (255, 255, 255), -1)
        cv2.circle(mini_map, (lg_line_x, gy_bot), 3, (255, 255, 255), -1)
        # Goal mouth opening bar (BRIGHT CYAN, thick)
        cv2.line(mini_map, (lg_line_x, gy_top), (lg_line_x, gy_bot), (0, 255, 255), 3)

        # ═══════════════════════════════════════════════════════
        # RIGHT GOAL (x=pitch_length side)
        # ═══════════════════════════════════════════════════════
        rg_line_x = pad + pitch_w_px  # goal line x position
        rg_net_x = rg_line_x + net_depth  # back of net

        trap_pts_r = np.array([
            [rg_line_x, gy_top],
            [rg_line_x, gy_bot],
            [rg_net_x, gy_bot - shrink],
            [rg_net_x, gy_top + shrink],
        ], dtype=np.int32)

        cv2.fillPoly(mini_map, [trap_pts_r], (50, 50, 50))
        for ny in range(gy_top + shrink, gy_bot - shrink, 3):
            cv2.line(mini_map, (rg_line_x, ny), (rg_net_x, ny), (120, 120, 120), 1)
        for nx in range(rg_line_x, rg_net_x, 3):
            frac_x = (nx - rg_line_x) / max(1, (rg_net_x - rg_line_x))
            top_y = int(gy_top + shrink * (1 - frac_x))
            bot_y = int(gy_bot - shrink * (1 - frac_x))
            cv2.line(mini_map, (nx, top_y), (nx, bot_y), (120, 120, 120), 1)
        cv2.polylines(mini_map, [trap_pts_r], True, (200, 200, 200), 1)
        cv2.circle(mini_map, (rg_line_x, gy_top), 3, (255, 255, 255), -1)
        cv2.circle(mini_map, (rg_line_x, gy_bot), 3, (255, 255, 255), -1)
        cv2.line(mini_map, (rg_line_x, gy_top), (rg_line_x, gy_bot), (0, 255, 255), 3)

        # ── Standard pitch markings ────────────────────────────
        # Pitch outline
        cv2.rectangle(mini_map, (pad, pad), (pad + pitch_w_px, pad + pitch_h_px), (255, 255, 255), 2)
        # Center line
        cx = pad + pitch_w_px // 2
        cv2.line(mini_map, (cx, pad), (cx, pad + pitch_h_px), (255, 255, 255), 1)
        # Center circle
        cv2.circle(mini_map, (cx, pad + pitch_h_px // 2), 20, (255, 255, 255), 1)
        # Center spot
        cv2.circle(mini_map, (cx, pad + pitch_h_px // 2), 2, (255, 255, 255), -1)

        # Penalty areas (scaled proportions)
        pa_w = int((16.5 / self.pitch_length) * pitch_w_px)
        pa_h = int((40.32 / self.pitch_width) * pitch_h_px)
        pa_y_offset = pad + (pitch_h_px - pa_h) // 2
        cv2.rectangle(mini_map, (pad, pa_y_offset), (pad + pa_w, pa_y_offset + pa_h), (255, 255, 255), 1)
        cv2.rectangle(mini_map, (pad + pitch_w_px - pa_w, pa_y_offset), (pad + pitch_w_px, pa_y_offset + pa_h), (255, 255, 255), 1)

        # 6-yard boxes
        sy_w = int((5.5 / self.pitch_length) * pitch_w_px)
        sy_h = int((18.32 / self.pitch_width) * pitch_h_px)
        sy_y = pad + (pitch_h_px - sy_h) // 2
        cv2.rectangle(mini_map, (pad, sy_y), (pad + sy_w, sy_y + sy_h), (255, 255, 255), 1)
        cv2.rectangle(mini_map, (pad + pitch_w_px - sy_w, sy_y), (pad + pitch_w_px, sy_y + sy_h), (255, 255, 255), 1)

        # ── Draw ball (with goal-in-net indicator) ─────────────
        ball_in_net = False
        if ball_pos_m is not None:
            bpx, bpy = m2px(ball_pos_m[0], ball_pos_m[1])
            # Check if ball is inside goal mouth
            in_goal_y = self.goal_y_min_m - 1.0 <= ball_pos_m[1] <= self.goal_y_max_m + 1.0
            in_left_net = ball_pos_m[0] <= 2.0 and in_goal_y
            in_right_net = ball_pos_m[0] >= (self.pitch_length - 2.0) and in_goal_y
            ball_in_net = in_left_net or in_right_net

            if ball_in_net:
                # Ball inside net: bright green pulsing dot with glow
                cv2.circle(mini_map, (bpx, bpy), 8, (0, 200, 0), 2)
                cv2.circle(mini_map, (bpx, bpy), 5, (0, 255, 0), -1)
                cv2.circle(mini_map, (bpx, bpy), 5, (255, 255, 255), 1)
            else:
                cv2.circle(mini_map, (bpx, bpy), 4, (0, 255, 255), -1)
                cv2.circle(mini_map, (bpx, bpy), 4, (0, 0, 0), 1)

        # ── Draw players ──────────────────────────────────────
        for t_id, (x_m, y_m) in metric_positions.items():
            px, py = m2px(x_m, y_m)

            team_id = team_assignments.get(t_id, 0)
            color = self.color_team_a if team_id == 0 else self.color_team_b

            radius = 5
            if t_id == possession_id:
                radius = 7
                cv2.circle(mini_map, (px, py), radius + 2, (0, 0, 255), 2)

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
