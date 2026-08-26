"""Tests for Stage D: Field Line Detector"""
import numpy as np
import pytest
from src.field_detector import FieldLineDetector


class TestFieldLineDetector:
    def test_empty_result(self):
        result = FieldLineDetector._empty_result()
        assert result['n_lines'] == 0
        assert result['n_circles'] == 0
        assert result['lines'] == []

    def test_detect_on_green_frame(self):
        """A uniform green frame should detect zero or very few lines."""
        detector = FieldLineDetector()
        frame = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)  # Green
        result = detector.detect_field_lines(frame)
        assert isinstance(result, dict)
        assert 'lines' in result
        # Uniform green = no lines
        assert result['n_lines'] == 0

    def test_detect_on_frame_with_white_line(self):
        """A green frame with a white horizontal line should detect at least one line."""
        detector = FieldLineDetector(min_line_length=30, hough_threshold=30)
        frame = np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8)
        # Draw a white horizontal line
        frame[350:355, 100:1100] = (255, 255, 255)
        result = detector.detect_field_lines(frame)
        assert result['n_lines'] >= 1

    def test_batch_detection(self):
        detector = FieldLineDetector()
        frames = [np.full((720, 1280, 3), (34, 139, 34), dtype=np.uint8) for _ in range(10)]
        results = detector.detect_field_lines_batch(frames, sample_every_n=3)
        assert len(results) == 10

    def test_classify_pitch_regions(self):
        detector = FieldLineDetector()
        field_data = {
            'horizontal_lines': [(100, 100, 1100, 100), (100, 600, 1100, 600)],
            'vertical_lines': [(640, 100, 640, 600)],
            'circles': [(640, 360, 50)]
        }
        classified = detector.classify_pitch_regions(field_data, 1280, 720)
        assert 'sidelines' in classified
        assert 'center_circle' in classified
        assert classified['center_circle'] is not None
