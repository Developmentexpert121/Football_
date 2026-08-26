import pytest
import numpy as np
from src.team_assigner import TeamAssigner

def test_team_assigner_clustering():
    assigner = TeamAssigner(n_clusters=2, mask_grass=False)
    
    # Red frame (Team A)
    red_frame = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)
    # Blue frame (Team B)
    blue_frame = np.full((200, 200, 3), (255, 0, 0), dtype=np.uint8)

    tracks_frame = [
        [{'track_id': 1, 'bbox': [10, 10, 100, 150], 'class_id': 0, 'conf': 0.9}],
        [{'track_id': 2, 'bbox': [10, 10, 100, 150], 'class_id': 0, 'conf': 0.9}]
    ]

    assigner.fit_team_colors([red_frame, blue_frame], tracks_frame)
    assert assigner.is_fitted is True

    team1 = assigner.get_player_team(red_frame, [10, 10, 100, 150], 1)
    team2 = assigner.get_player_team(blue_frame, [10, 10, 100, 150], 2)
    assert team1 in (0, 1)
    assert team2 in (0, 1)
    assert team1 != team2, "Distinct jersey colors should yield distinct team cluster IDs."
