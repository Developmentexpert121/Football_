import cv2
import numpy as np
from typing import List, Tuple, Optional

class HomographyTransformer:
    """
    Stage 7: Maps image pixel coordinates (x, y) to real-world pitch metric coordinates (X_m, Y_m).
    Standard pitch dimensions: 105.0 meters x 68.0 meters.
    """
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        ref_image_points: Optional[List[List[float]]] = None,
        ref_pitch_points: Optional[List[List[float]]] = None
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Default standard perspective mapping reference points
        if ref_image_points is None:
            self.ref_image_points = np.array([
                [200.0, 150.0],   # Top-Left corner
                [1080.0, 150.0],  # Top-Right corner
                [1200.0, 680.0],  # Bottom-Right corner
                [80.0, 680.0]     # Bottom-Left corner
            ], dtype=np.float32)
        else:
            self.ref_image_points = np.array(ref_image_points, dtype=np.float32)

        if ref_pitch_points is None:
            self.ref_pitch_points = np.array([
                [0.0, 0.0],
                [pitch_length, 0.0],
                [pitch_length, pitch_width],
                [0.0, pitch_width]
            ], dtype=np.float32)
        else:
            self.ref_pitch_points = np.array(ref_pitch_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(self.ref_image_points, self.ref_pitch_points)

    def transform_point(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transforms a single pixel coordinate (x, y) to metric pitch position (x_m, y_m).
        """
        pts = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H)
        x_m = float(transformed[0][0][0])
        y_m = float(transformed[0][0][1])

        # Clamp inside pitch metric bounds
        x_m = max(0.0, min(self.pitch_length, x_m))
        y_m = max(0.0, min(self.pitch_width, y_m))
        return (x_m, y_m)

    def transform_bbox_bottom(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Extracts bottom-center point of bounding box [x1, y1, x2, y2] and transforms to pitch meters.
        """
        bottom_center_x = (bbox[0] + bbox[2]) / 2.0
        bottom_center_y = bbox[3]
        return self.transform_point((bottom_center_x, bottom_center_y))
