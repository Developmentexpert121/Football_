import cv2
import os
import pickle
import numpy as np
from typing import List, Tuple

class CameraMovementEstimator:
    """
    Stage 6: Estimates frame-to-frame camera movement (pan, tilt, zoom)
    using Lucas-Kanade Optical Flow on pitch background keypoints.
    """
    def __init__(self, stub_path: str = "stubs/camera_movement.pkl"):
        self.stub_path = stub_path

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Returns list of (dx, dy) optical flow offset per frame relative to previous frame.
        """
        if read_from_stub and os.path.exists(self.stub_path):
            try:
                with open(self.stub_path, "rb") as f:
                    stub_data = pickle.load(f)
                    if isinstance(stub_data, list) and len(stub_data) == len(frames):
                        print(f"Loading cached camera movement from stub: {self.stub_path}")
                        return stub_data
                    else:
                        print(f"Stub camera movement frame count ({len(stub_data) if isinstance(stub_data, list) else 'invalid'}) does not match input frame count ({len(frames)}). Recomputing camera movement...")
            except Exception as e:
                print(f"Error reading camera movement stub cache ({e}). Recomputing camera movement...")

        camera_movement = [(0.0, 0.0)]
        if len(frames) < 2:
            return camera_movement

        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Find corners on pitch background
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) > 0:
                        movement = good_new - good_old
                        dx = float(np.median(movement[:, 0]))
                        dy = float(np.median(movement[:, 1]))
                    else:
                        dx, dy = 0.0, 0.0
                else:
                    dx, dy = 0.0, 0.0
            else:
                dx, dy = 0.0, 0.0

            camera_movement.append((dx, dy))
            prev_gray = curr_gray

        os.makedirs(os.path.dirname(self.stub_path), exist_ok=True)
        with open(self.stub_path, "wb") as f:
            pickle.dump(camera_movement, f)

        return camera_movement
