"""Tests for Stage A: Ball Tracker (Kalman Filter)"""
import numpy as np
import pytest
from src.ball_tracker import KalmanBallTracker, smooth_ball_trajectory


class TestKalmanBallTracker:
    def test_init_on_first_detection(self):
        kf = KalmanBallTracker()
        result = kf.update(100.0, 200.0)
        assert result is not None
        assert abs(result[0] - 100.0) < 1e-3
        assert abs(result[1] - 200.0) < 1e-3

    def test_returns_none_before_first_detection(self):
        kf = KalmanBallTracker()
        result = kf.update(None, None)
        assert result is None

    def test_predicts_through_occlusion(self):
        kf = KalmanBallTracker(max_lost_frames=10)
        # Give two detections to establish velocity
        kf.update(100.0, 100.0)
        kf.update(110.0, 100.0)
        # Now lose detection for 3 frames
        r1 = kf.update(None, None)
        r2 = kf.update(None, None)
        r3 = kf.update(None, None)
        # Should still have predictions (moving right)
        assert r1 is not None
        assert r2 is not None
        assert r3 is not None
        assert r3[0] > r1[0]  # Ball should continue moving right

    def test_resets_after_max_lost(self):
        kf = KalmanBallTracker(max_lost_frames=3)
        kf.update(100.0, 100.0)
        for _ in range(5):
            result = kf.update(None, None)
        assert result is None

    def test_smooth_ball_trajectory(self):
        # Simulate tracks with ball class_id=3
        tracks = [
            [{'class_id': 3, 'bbox': [95, 95, 105, 105], 'track_id': 99}],
            [{'class_id': 3, 'bbox': [105, 95, 115, 105], 'track_id': 99}],
            [{'class_id': 0, 'bbox': [0, 0, 10, 10], 'track_id': 1}],  # No ball
            [{'class_id': 3, 'bbox': [125, 95, 135, 105], 'track_id': 99}],
        ]
        result = smooth_ball_trajectory(tracks, max_lost_frames=5)
        assert len(result) == 4
        assert result[0] is not None
        assert result[2] is not None  # Should interpolate
        assert result[2]['interpolated'] is True
