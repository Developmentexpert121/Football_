"""Tests for Enhanced Analytics Engine (Stage E)"""
import numpy as np
import pytest
from src.analytics_engine import AnalyticsEngine


class TestAnalyticsEngineEnhanced:
    def _make_test_data(self, n_frames=30, n_players=4):
        """Generate synthetic tracking data for analytics testing."""
        tracks_per_frame = []
        positions_per_frame = []
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        for frame in range(n_frames):
            tracks = []
            pos_dict = {}

            for pid in range(1, n_players + 1):
                x = 20.0 + pid * 10.0 + frame * 0.5  # Moving right
                y = 30.0 + np.sin(frame * 0.3) * 5
                tracks.append({'track_id': pid, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9})
                pos_dict[pid] = (x, y)

            # Add ball
            ball_x = 50.0 + frame * 0.8
            ball_y = 34.0
            tracks.append({'track_id': 99, 'class_id': 3, 'bbox': [0, 0, 5, 5], 'conf': 0.95})
            pos_dict[99] = (ball_x, ball_y)

            tracks_per_frame.append(tracks)
            positions_per_frame.append(pos_dict)

        return tracks_per_frame, positions_per_frame, team_assignments

    def test_compute_returns_all_keys(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data()
        result = engine.compute_analytics(tracks, positions, teams)
        
        assert 'player_stats' in result
        assert 'possession_stats' in result
        assert 'ball_trajectories' in result
        assert 'speeds_per_frame' in result
        assert 'ball_metric_per_frame' in result
        assert 'touch_counts' in result
        assert 'pass_counts' in result
        assert 'team_heatmaps' in result
        assert 'pressure_data' in result

    def test_player_stats_have_new_metrics(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data()
        result = engine.compute_analytics(tracks, positions, teams)
        
        for pid, stats in result['player_stats'].items():
            assert 'sprint_count' in stats
            assert 'sprint_time_sec' in stats
            assert 'avg_acceleration_ms2' in stats
            assert 'max_acceleration_ms2' in stats
            assert 'max_deceleration_ms2' in stats
            assert 'high_intensity_changes' in stats
            assert 'avg_running_direction' in stats
            assert 'touch_count' in stats
            assert 'pass_count' in stats

    def test_possession_percentages_valid(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data()
        result = engine.compute_analytics(tracks, positions, teams)
        
        poss = result['possession_stats']
        assert poss['team_a_possession_pct'] >= 0
        assert poss['team_b_possession_pct'] >= 0
        assert poss['team_a_possession_pct'] + poss['team_b_possession_pct'] <= 100.5

    def test_speeds_per_frame_length(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data(n_frames=20)
        result = engine.compute_analytics(tracks, positions, teams)
        assert len(result['speeds_per_frame']) == 20

    def test_ball_metric_per_frame_length(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data(n_frames=20)
        result = engine.compute_analytics(tracks, positions, teams)
        assert len(result['ball_metric_per_frame']) == 20

    def test_pressure_data(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data()
        result = engine.compute_analytics(tracks, positions, teams)
        pressure = result['pressure_data']
        assert 'avg_pressure' in pressure
        assert 'max_pressure' in pressure

    def test_team_heatmaps(self):
        engine = AnalyticsEngine(fps=25.0)
        tracks, positions, teams = self._make_test_data()
        result = engine.compute_analytics(tracks, positions, teams)
        heatmaps = result['team_heatmaps']
        assert 0 in heatmaps
        assert 1 in heatmaps
        assert len(heatmaps[0]) > 0
        assert len(heatmaps[1]) > 0
