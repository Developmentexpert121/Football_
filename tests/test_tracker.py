import pytest
import numpy as np
from src.tracker import MultiObjectTracker

def test_tracker_id_persistence():
    tracker = MultiObjectTracker(track_high_thresh=0.3)
    frame1_dets = [{'bbox': [100, 100, 150, 200], 'class_id': 0, 'conf': 0.9}]
    frame2_dets = [{'bbox': [102, 101, 152, 201], 'class_id': 0, 'conf': 0.88}]

    frames = [np.zeros((720, 1280, 3), dtype=np.uint8)] * 2
    tracks = tracker.track_frames(frames, [frame1_dets, frame2_dets], read_from_stub=False)

    assert len(tracks) == 2
    id_f1 = tracks[0][0]['track_id']
    id_f2 = tracks[1][0]['track_id']
    assert id_f1 == id_f2, "Tracker should maintain identical ID across consecutive overlapping bounding boxes."
