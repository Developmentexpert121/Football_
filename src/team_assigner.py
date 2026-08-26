import cv2
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

class TeamAssigner:
    """
    Stage 5: Clusters players into Team 0 and Team 1 using K-Means clustering
    on central chest jersey crops in Lab/HSV color space with green pitch grass masking.
    Features majority voting across frames and dynamic team color detection.
    """
    def __init__(self, n_clusters: int = 2, mask_grass: bool = True):
        self.n_clusters = n_clusters
        self.mask_grass = mask_grass
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42) if SKLEARN_AVAILABLE else None
        # Default team BGR colors (Team 0: Red, Team 1: Blue/White)
        self.team_colors: Dict[int, Tuple[int, int, int]] = {0: (50, 50, 255), 1: (255, 100, 50)}
        self.team_names: Dict[int, str] = {0: "Team Red", 1: "Team White"}
        self.player_team_dict: Dict[int, int] = {}
        self.player_samples: Dict[int, List[int]] = defaultdict(list)
        self.is_fitted = False

    def fit_team_colors(self, frames: List[np.ndarray], tracks_per_frame: List[List[Dict[str, Any]]]):
        """
        Extracts chest jersey crops across frames and fits K-Means model.
        Also computes dynamic team colors and readable team names.
        """
        player_colors = []
        player_bgr_colors = []

        for frame_idx, tracks in enumerate(tracks_per_frame):
            frame = frames[frame_idx]
            for track in tracks:
                if track['class_id'] == 0:  # Player
                    bgr_color = self._extract_chest_color(frame, track['bbox'])
                    if bgr_color is not None:
                        # Convert BGR to Lab for distance clustering
                        bgr_pixel = np.uint8([[bgr_color]])
                        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0][0]
                        player_colors.append(lab_pixel)
                        player_bgr_colors.append(bgr_color)

        if len(player_colors) < self.n_clusters or not SKLEARN_AVAILABLE:
            self.team_colors = {0: (50, 50, 255), 1: (240, 240, 240)}
            self.team_names = {0: "Team Red", 1: "Team White"}
            self.is_fitted = True
            return

        X = np.array(player_colors)
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(X)

        # Compute dynamic average BGR color per team cluster
        bgr_array = np.array(player_bgr_colors)
        for cluster_id in range(self.n_clusters):
            cluster_mask = (labels == cluster_id)
            if np.any(cluster_mask):
                mean_bgr = np.mean(bgr_array[cluster_mask], axis=0)
                b, g, r = map(int, mean_bgr)
                self.team_colors[cluster_id] = (b, g, r)
            else:
                self.team_colors[cluster_id] = (50, 50, 255) if cluster_id == 0 else (240, 240, 240)

        # Assign human-readable names based on dominant colors
        self._name_teams_from_colors()

        # Fit majority votes for all players across all sampled frames
        for frame_idx, tracks in enumerate(tracks_per_frame):
            frame = frames[frame_idx]
            for track in tracks:
                if track['class_id'] == 0:
                    t_id = track['track_id']
                    bgr_color = self._extract_chest_color(frame, track['bbox'])
                    if bgr_color is not None and self.kmeans is not None:
                        bgr_pixel = np.uint8([[bgr_color]])
                        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0][0]
                        pred_team = int(self.kmeans.predict([lab_pixel])[0])
                        self.player_samples[t_id].append(pred_team)

        # Resolve final team for each player via majority voting
        for p_id, samples in self.player_samples.items():
            if samples:
                most_common = Counter(samples).most_common(1)[0][0]
                self.player_team_dict[p_id] = most_common

        self.is_fitted = True

    def get_player_team(self, frame: np.ndarray, bbox: List[float], player_id: int) -> int:
        """
        Returns team label (0 or 1) for a player using majority voting or real-time inference.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if not self.is_fitted or self.kmeans is None:
            team_id = player_id % self.n_clusters
            self.player_team_dict[player_id] = team_id
            return team_id

        bgr_color = self._extract_chest_color(frame, bbox)
        if bgr_color is None:
            team_id = player_id % self.n_clusters
        else:
            bgr_pixel = np.uint8([[bgr_color]])
            lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0][0]
            team_id = int(self.kmeans.predict([lab_pixel])[0])

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_team_color(self, team_id: int) -> Tuple[int, int, int]:
        """Returns BGR color tuple for a team."""
        return self.team_colors.get(team_id, (50, 50, 255) if team_id == 0 else (255, 100, 50))

    def get_team_name(self, team_id: int) -> str:
        """Returns human-readable name for a team (e.g. 'Team Red', 'Team White')."""
        return self.team_names.get(team_id, f"Team {team_id + 1}")

    def _name_teams_from_colors(self):
        """Generates friendly team names based on cluster BGR color profile."""
        for team_id, bgr in self.team_colors.items():
            b, g, r = bgr
            if r > 160 and g > 160 and b > 160:
                self.team_names[team_id] = "Team White"
            elif r > 150 and r > g + 40 and r > b + 40:
                self.team_names[team_id] = "Team Red"
            elif b > 150 and b > r + 30:
                self.team_names[team_id] = "Team Blue"
            elif g > 150 and g > r + 30:
                self.team_names[team_id] = "Team Green"
            elif r > 150 and g > 150 and b < 100:
                self.team_names[team_id] = "Team Yellow"
            else:
                self.team_names[team_id] = "Team A" if team_id == 0 else "Team B"
                
        # Ensure symmetric naming scheme
        names = list(self.team_names.values())
        if "Team A" in names or "Team B" in names:
            self.team_names[0] = "Team A"
            self.team_names[1] = "Team B"

    def _extract_chest_color(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Crops central chest region (15-45% height, 20-80% width) to avoid background, skin, and shorts.
        Filters pitch green grass pixels in HSV space and returns median BGR color.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)

        # Central chest ROI
        chest_y1 = max(0, y1 + int(0.15 * h))
        chest_y2 = min(frame.shape[0], y1 + int(0.48 * h))
        chest_x1 = max(0, x1 + int(0.20 * w))
        chest_x2 = min(frame.shape[1], x1 + int(0.80 * w))

        chest = frame[chest_y1:chest_y2, chest_x1:chest_x2]
        if chest.size == 0:
            return None

        if self.mask_grass:
            hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
            # Mask out grass (Hue 32..88, Saturation > 30)
            grass_mask = cv2.inRange(hsv, (32, 30, 30), (88, 255, 255))
            non_grass = chest[grass_mask == 0]
            if len(non_grass) > 10:
                return np.median(non_grass, axis=0)

        # Fallback to median of entire chest crop
        return np.median(chest.reshape(-1, 3), axis=0)

