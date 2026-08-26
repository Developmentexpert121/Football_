"""Tests for Stage G: Tactical Analyzer"""
import numpy as np
import pytest
from src.tactical_analyzer import TacticalAnalyzer


class TestTacticalAnalyzer:
    def _make_tracks_and_positions(self, n_frames=20, n_players_per_team=5):
        """Generate synthetic tracked data for testing."""
        tracks_per_frame = []
        positions_per_frame = []
        team_assignments = {}

        # Team 0: players 1-5 on left half, Team 1: players 6-10 on right half
        for pid in range(1, n_players_per_team + 1):
            team_assignments[pid] = 0
        for pid in range(n_players_per_team + 1, 2 * n_players_per_team + 1):
            team_assignments[pid] = 1

        # Ball ID
        ball_id = 99
        ball_positions = []

        for frame in range(n_frames):
            tracks = []
            pos_dict = {}

            for pid in range(1, n_players_per_team + 1):
                x = 20.0 + pid * 8.0 + np.random.randn() * 2
                y = 15.0 + pid * 8.0 + np.random.randn() * 2
                tracks.append({'track_id': pid, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9})
                pos_dict[pid] = (x, y)

            for pid in range(n_players_per_team + 1, 2 * n_players_per_team + 1):
                x = 55.0 + (pid - n_players_per_team) * 8.0 + np.random.randn() * 2
                y = 15.0 + (pid - n_players_per_team) * 8.0 + np.random.randn() * 2
                tracks.append({'track_id': pid, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9})
                pos_dict[pid] = (x, y)

            # Ball moves across pitch
            ball_x = 20.0 + frame * 3.0
            ball_y = 34.0 + np.sin(frame * 0.5) * 10
            tracks.append({'track_id': ball_id, 'class_id': 3, 'bbox': [0, 0, 5, 5], 'conf': 0.95})
            pos_dict[ball_id] = (ball_x, ball_y)
            ball_positions.append((ball_x, ball_y))

            tracks_per_frame.append(tracks)
            positions_per_frame.append(pos_dict)

        return tracks_per_frame, positions_per_frame, team_assignments, ball_positions

    def test_analyze_returns_all_keys(self):
        analyzer = TacticalAnalyzer()
        tracks, positions, teams, ball_pos = self._make_tracks_and_positions()
        result = analyzer.analyze(tracks, positions, teams, ball_metric_per_frame=ball_pos)
        
        assert 'formations' in result
        assert 'defensive_line_height' in result
        assert 'pressing_intensity' in result
        assert 'attacking_zones' in result
        assert 'passing_network' in result
        assert 'touch_counts' in result
        assert 'compactness' in result
        assert 'width_depth' in result

    def test_formations_detected(self):
        analyzer = TacticalAnalyzer()
        tracks, positions, teams, ball_pos = self._make_tracks_and_positions()
        result = analyzer.analyze(tracks, positions, teams, ball_metric_per_frame=ball_pos)
        assert 'team_0' in result['formations']
        assert 'team_1' in result['formations']

    def test_attacking_zones_sum_to_100(self):
        analyzer = TacticalAnalyzer()
        tracks, positions, teams, ball_pos = self._make_tracks_and_positions(n_frames=50)
        result = analyzer.analyze(tracks, positions, teams, ball_metric_per_frame=ball_pos)
        zones = result['attacking_zones']
        total = zones['defensive_third_pct'] + zones['midfield_third_pct'] + zones['attacking_third_pct']
        assert abs(total - 100.0) < 1.0

    def test_report_text_generation(self):
        analyzer = TacticalAnalyzer()
        tracks, positions, teams, ball_pos = self._make_tracks_and_positions()
        result = analyzer.analyze(tracks, positions, teams, ball_metric_per_frame=ball_pos)
        report = analyzer.generate_tactical_report_text(result)
        assert 'TACTICAL ANALYSIS REPORT' in report
        assert 'Formation' in report
