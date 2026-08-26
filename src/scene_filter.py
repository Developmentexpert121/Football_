import cv2
import numpy as np
from typing import List, Tuple

class SceneFilter:
    """
    Stage 2: Filters out non-action frames (replays, crowd shots, close-up interviews)
    based on the ratio of green pitch grass pixels in HSV color space.
    """
    def __init__(
        self,
        green_ratio_threshold: float = 0.35,
        hsv_min: Tuple[int, int, int] = (35, 40, 40),
        hsv_max: Tuple[int, int, int] = (85, 255, 255)
    ):
        self.green_ratio_threshold = green_ratio_threshold
        self.hsv_min = np.array(hsv_min, dtype=np.uint8)
        self.hsv_max = np.array(hsv_max, dtype=np.uint8)

    def compute_green_ratio(self, frame: np.ndarray) -> float:
        """
        Calculates the proportion of green pixels in the frame.
        """
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.hsv_min, self.hsv_max)
        green_pixel_count = np.count_nonzero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        return float(green_pixel_count / total_pixels)

    def is_action_frame(self, frame: np.ndarray) -> bool:
        """
        Returns True if the frame contains sufficient pitch grass to be considered an active play frame.
        """
        return self.compute_green_ratio(frame) >= self.green_ratio_threshold

    def filter_frames(self, frames: List[np.ndarray]) -> List[Tuple[int, bool, float]]:
        """
        Evaluates a list of frames. Returns list of tuples: (frame_index, is_action, green_ratio).
        """
        results = []
        for idx, frame in enumerate(frames):
            ratio = self.compute_green_ratio(frame)
            is_action = ratio >= self.green_ratio_threshold
            results.append((idx, is_action, ratio))
        return results
