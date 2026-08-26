"""Tests for Stage F: Action Recognizer"""
import numpy as np
import pytest
from src.action_recognizer import (
    ActionRecognizer,
    ACTION_STANDING, ACTION_WALKING, ACTION_RUNNING, ACTION_SPRINTING,
    ACTION_KICKING, ACTION_SLIDING_TACKLE, ACTION_HEADER, ACTION_DRIBBLING
)


class TestActionRecognizer:
    def _make_pose_data(self, **kp_overrides):
        """Create mock keypoints (17, 3) with reasonable defaults."""
        kps = np.zeros((17, 3))
        # Default upright human pose
        kps[0] = [100, 50, 0.9]    # nose
        kps[5] = [80, 100, 0.9]    # left shoulder
        kps[6] = [120, 100, 0.9]   # right shoulder
        kps[11] = [85, 200, 0.9]   # left hip
        kps[12] = [115, 200, 0.9]  # right hip
        kps[13] = [85, 300, 0.9]   # left knee
        kps[14] = [115, 300, 0.9]  # right knee
        kps[15] = [85, 400, 0.9]   # left ankle
        kps[16] = [115, 400, 0.9]  # right ankle
        for key, val in kp_overrides.items():
            idx = int(key.split('_')[1])
            kps[idx] = val
        return kps

    def test_speed_based_standing(self):
        ar = ActionRecognizer()
        pose = [{1: {'keypoints': self._make_pose_data()}}]
        speeds = [{1: 0.5}]
        result = ar.classify_actions(pose, speeds)
        assert result[0][1] == ACTION_STANDING

    def test_speed_based_walking(self):
        ar = ActionRecognizer()
        pose = [{1: {'keypoints': self._make_pose_data()}}]
        speeds = [{1: 5.0}]
        result = ar.classify_actions(pose, speeds)
        assert result[0][1] == ACTION_WALKING

    def test_speed_based_running(self):
        ar = ActionRecognizer()
        pose = [{1: {'keypoints': self._make_pose_data()}}]
        speeds = [{1: 15.0}]
        result = ar.classify_actions(pose, speeds)
        assert result[0][1] == ACTION_RUNNING

    def test_speed_based_sprinting(self):
        ar = ActionRecognizer()
        pose = [{1: {'keypoints': self._make_pose_data()}}]
        speeds = [{1: 28.0}]
        result = ar.classify_actions(pose, speeds)
        assert result[0][1] == ACTION_SPRINTING

    def test_sliding_tackle_detection(self):
        """Hips close to ankle height = sliding tackle."""
        ar = ActionRecognizer()
        kps = self._make_pose_data()
        # Move hips down near ankle level
        kps[11] = [85, 385, 0.9]
        kps[12] = [115, 385, 0.9]
        pose = [{1: {'keypoints': kps}}]
        speeds = [{1: 8.0}]
        result = ar.classify_actions(pose, speeds)
        assert result[0][1] == ACTION_SLIDING_TACKLE

    def test_header_detection(self):
        """Ball near head at shoulder height."""
        ar = ActionRecognizer()
        kps = self._make_pose_data()
        pose = [{1: {'keypoints': kps}}]
        speeds = [{1: 5.0}]
        ball_px = [(100, 40)]  # Near nose and above shoulders
        result = ar.classify_actions(pose, speeds, ball_positions_per_frame=ball_px)
        assert result[0][1] == ACTION_HEADER

    def test_dribbling_detection(self):
        """Running + near ball = dribbling."""
        ar = ActionRecognizer()
        kps = self._make_pose_data()
        pose = [{1: {'keypoints': kps}}]
        speeds = [{1: 12.0}]
        metric_pos = [{1: (50.0, 30.0)}]
        ball_m = [(50.5, 30.0)]
        result = ar.classify_actions(
            pose, speeds,
            metric_positions_per_frame=metric_pos,
            ball_metric_per_frame=ball_m
        )
        assert result[0][1] == ACTION_DRIBBLING

    def test_empty_pose_data(self):
        ar = ActionRecognizer()
        result = ar.classify_actions([{}], [{}])
        assert result[0] == {}
