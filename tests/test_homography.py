import pytest
from src.homography import HomographyTransformer

def test_homography_transform_bounds():
    transformer = HomographyTransformer(pitch_length=105.0, pitch_width=68.0)
    
    # Test center pixel transform
    x_m, y_m = transformer.transform_point((640, 360))
    assert 0.0 <= x_m <= 105.0
    assert 0.0 <= y_m <= 68.0

def test_homography_bbox_transform():
    transformer = HomographyTransformer()
    bbox = [100, 100, 200, 500]
    x_m, y_m = transformer.transform_bbox_bottom(bbox)
    assert isinstance(x_m, float)
    assert isinstance(y_m, float)
