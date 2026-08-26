"""Tests for Stage C: Pose Estimator"""
import numpy as np
import pytest
from src.pose_estimator import PoseEstimator, KEYPOINT_NAMES, SKELETON_EDGES


class TestPoseEstimatorUtils:
    def test_keypoint_names_length(self):
        assert len(KEYPOINT_NAMES) == 17

    def test_skeleton_edges_valid(self):
        for start, end in SKELETON_EDGES:
            assert 0 <= start < 17
            assert 0 <= end < 17

    def test_get_keypoint_valid(self):
        kps = np.zeros((17, 3))
        kps[0] = [100.0, 200.0, 0.9]  # nose
        result = PoseEstimator.get_keypoint(kps, 'nose')
        assert result is not None
        assert abs(result[0] - 100.0) < 1e-3
        assert abs(result[1] - 200.0) < 1e-3

    def test_get_keypoint_low_confidence(self):
        kps = np.zeros((17, 3))
        kps[0] = [100.0, 200.0, 0.1]  # Low confidence
        result = PoseEstimator.get_keypoint(kps, 'nose')
        assert result is None

    def test_get_keypoint_invalid_name(self):
        kps = np.zeros((17, 3))
        result = PoseEstimator.get_keypoint(kps, 'nonexistent')
        assert result is None

    def test_joint_angle_straight(self):
        """Test 180-degree angle (straight line)"""
        kps = np.zeros((17, 3))
        kps[11] = [0, 0, 1.0]    # left hip
        kps[13] = [0, 50, 1.0]   # left knee
        kps[15] = [0, 100, 1.0]  # left ankle
        angle = PoseEstimator.get_joint_angle(kps, 11, 13, 15)
        assert abs(angle - 180.0) < 1.0

    def test_joint_angle_right_angle(self):
        """Test 90-degree angle"""
        kps = np.zeros((17, 3))
        kps[11] = [0, 0, 1.0]
        kps[13] = [0, 50, 1.0]
        kps[15] = [50, 50, 1.0]
        angle = PoseEstimator.get_joint_angle(kps, 11, 13, 15)
        assert abs(angle - 90.0) < 1.0

    def test_iou_identical_boxes(self):
        iou = PoseEstimator._compute_iou([0, 0, 10, 10], [0, 0, 10, 10])
        assert abs(iou - 1.0) < 1e-3

    def test_iou_no_overlap(self):
        iou = PoseEstimator._compute_iou([0, 0, 10, 10], [20, 20, 30, 30])
        assert abs(iou) < 1e-3
