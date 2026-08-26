import pytest
import numpy as np
from src.scene_filter import SceneFilter

def test_scene_filter_green_pitch():
    filter_obj = SceneFilter(green_ratio_threshold=0.35)
    # Create pure green frame (HSV green)
    green_frame = np.full((100, 100, 3), (34, 139, 34), dtype=np.uint8)
    ratio = filter_obj.compute_green_ratio(green_frame)
    assert ratio > 0.35
    assert filter_obj.is_action_frame(green_frame) is True

def test_scene_filter_non_green():
    filter_obj = SceneFilter(green_ratio_threshold=0.35)
    # Create pure blue frame (crowd / non-pitch)
    blue_frame = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    ratio = filter_obj.compute_green_ratio(blue_frame)
    assert ratio < 0.35
    assert filter_obj.is_action_frame(blue_frame) is False
