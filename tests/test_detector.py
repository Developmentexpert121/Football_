import pytest
import numpy as np
from src.detector import ObjectDetector

def test_detector_output_format():
    detector = ObjectDetector(conf_thresh=0.1)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect_frame(frame)

    assert isinstance(detections, list)
    if len(detections) > 0:
        det = detections[0]
        assert 'bbox' in det
        assert 'class_id' in det
        assert 'conf' in det
        assert len(det['bbox']) == 4
