"""Tests for Enhanced Event Detector (Stage H)"""
import numpy as np
import pytest
from src.event_detector import EventDetector


class TestEventDetector:
    def test_foul_detection(self):
        """Two opposing players close together should trigger foul."""
        detector = EventDetector(fps=25.0, foul_dist_thresh=1.5)
        tracks = [[]] + [
            [
                {'track_id': 1, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9},
                {'track_id': 2, 'class_id': 0, 'bbox': [5, 5, 15, 15], 'conf': 0.9},
            ]
        ] * 3
        positions = [{}] + [
            {1: (50.0, 30.0), 2: (50.5, 30.0)}
        ] * 3
        team_assignments = {1: 0, 2: 1}
        events = detector.detect_events(tracks, positions, team_assignments)
        foul_events = [e for e in events if e['event_type'] == 'Potential Foul']
        assert len(foul_events) >= 1

    def test_goal_detection_right(self):
        """Ball near right goal line within goal mouth."""
        detector = EventDetector(fps=25.0, pitch_length=105.0)
        tracks = [[]] + [
            [{'track_id': 99, 'class_id': 3, 'bbox': [0, 0, 5, 5], 'conf': 0.9}]
        ] * 3
        positions = [{}] + [
            {99: (104.0, 34.0)}  # Near right goal, centered
        ] * 3
        team_assignments = {}
        ball_metric = [None] + [(104.0, 34.0)] * 3
        events = detector.detect_events(tracks, positions, team_assignments, ball_metric_per_frame=ball_metric)
        goal_events = [e for e in events if e['event_type'] == 'Goal']
        assert len(goal_events) >= 1

    def test_corner_kick_detection(self):
        """Ball near a corner flag."""
        detector = EventDetector(fps=25.0, corner_area_radius=5.0)
        tracks = [[]] + [
            [{'track_id': 99, 'class_id': 3, 'bbox': [0, 0, 5, 5], 'conf': 0.9}]
        ] * 3
        positions = [{}] + [
            {99: (2.0, 2.0)}  # Near top-left corner
        ] * 3
        team_assignments = {}
        ball_metric = [None] + [(2.0, 2.0)] * 3
        events = detector.detect_events(tracks, positions, team_assignments, ball_metric_per_frame=ball_metric)
        corner_events = [e for e in events if e['event_type'] == 'Corner Kick']
        assert len(corner_events) >= 1

    def test_deduplication(self):
        """Same event in consecutive frames should be deduplicated."""
        detector = EventDetector(fps=25.0, foul_dist_thresh=1.5)
        # Create many frames with the same foul proximity
        n_frames = 20
        tracks = [[]] + [
            [
                {'track_id': 1, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9},
                {'track_id': 2, 'class_id': 0, 'bbox': [5, 5, 15, 15], 'conf': 0.9},
            ]
        ] * (n_frames - 1)
        positions = [{}] + [
            {1: (50.0, 30.0), 2: (50.5, 30.0)}
        ] * (n_frames - 1)
        team_assignments = {1: 0, 2: 1}
        events = detector.detect_events(tracks, positions, team_assignments)
        foul_events = [e for e in events if e['event_type'] == 'Potential Foul']
        # Should be deduplicated to 1 or very few
        assert len(foul_events) <= 3

    def test_penalty_area_entry(self):
        """Ball + attacker inside penalty area."""
        detector = EventDetector(fps=25.0)
        tracks = [[]] + [
            [
                {'track_id': 1, 'class_id': 0, 'bbox': [0, 0, 10, 10], 'conf': 0.9},
                {'track_id': 99, 'class_id': 3, 'bbox': [0, 0, 5, 5], 'conf': 0.9},
            ]
        ] * 3
        # Player and ball in right penalty area
        positions = [{}] + [
            {1: (95.0, 34.0), 99: (96.0, 34.0)}
        ] * 3
        team_assignments = {1: 0}
        ball_metric = [None] + [(96.0, 34.0)] * 3
        events = detector.detect_events(tracks, positions, team_assignments, ball_metric_per_frame=ball_metric)
        pen_events = [e for e in events if e['event_type'] == 'Penalty Area Entry']
        assert len(pen_events) >= 1

    def test_no_events_on_empty_data(self):
        detector = EventDetector()
        events = detector.detect_events([], [], {})
        assert events == []
