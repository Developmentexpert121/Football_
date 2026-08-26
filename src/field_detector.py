"""
Stage D: Field Line Detection using OpenCV.

Detects pitch markings from broadcast-view frames:
- Sidelines, goal lines
- Penalty box borders (16.5m / 18-yard box)
- Center line and center circle
- Corner arcs

Uses Canny Edge Detection + Probabilistic Hough Lines (HoughLinesP).
No deep learning model required — runs entirely on OpenCV + NumPy.

Provides detected lines as anchor points for dynamic homography refinement.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class FieldLineDetector:
    """
    Detects football pitch lines and markings from broadcast-view video frames.

    Uses color-space masking to isolate field markings (white lines on green pitch),
    then applies Canny edge detection and Probabilistic Hough Transform to
    extract line segments.

    Results are classified into horizontal/vertical lines and circles.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 80,
        min_line_length: int = 80,
        max_line_gap: int = 30,
        circle_dp: float = 1.2,
        circle_min_dist: int = 100,
        circle_param1: int = 100,
        circle_param2: int = 40,
        circle_min_radius: int = 15,
        circle_max_radius: int = 120
    ):
        # Canny edge detection parameters
        self.canny_low = canny_low
        self.canny_high = canny_high

        # Hough line transform parameters
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # Circle detection parameters (for center circle)
        self.circle_dp = circle_dp
        self.circle_min_dist = circle_min_dist
        self.circle_param1 = circle_param1
        self.circle_param2 = circle_param2
        self.circle_min_radius = circle_min_radius
        self.circle_max_radius = circle_max_radius

        # HSV range for white line detection on green pitch
        # White lines appear as high-value, low-saturation pixels
        self.white_hsv_low = np.array([0, 0, 180])
        self.white_hsv_high = np.array([180, 60, 255])

        # Green pitch masking (for reference)
        self.green_hsv_low = np.array([35, 40, 40])
        self.green_hsv_high = np.array([85, 255, 255])

    def detect_field_lines(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detects all field line segments and circles in a single frame.

        Args:
            frame: BGR video frame (numpy array)

        Returns:
            Dict with keys:
                'lines': list of line segments [(x1,y1,x2,y2), ...]
                'horizontal_lines': lines close to horizontal (±15°)
                'vertical_lines': lines close to vertical (±15°)
                'diagonal_lines': all other lines
                'circles': list of detected circles [(cx, cy, radius), ...]
                'field_mask': binary mask of detected pitch area
                'line_mask': binary mask of detected lines
        """
        h, w = frame.shape[:2]

        # Step 1: Convert to HSV and isolate white markings
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_hsv_low, self.white_hsv_high)

        # Step 2: Also create green pitch mask (to filter lines outside the pitch)
        green_mask = cv2.inRange(hsv, self.green_hsv_low, self.green_hsv_high)
        # Dilate green mask to include nearby lines at pitch edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        field_region = cv2.dilate(green_mask, kernel_dilate, iterations=2)

        # Step 3: Only keep white pixels that are on/near the pitch
        line_candidates = cv2.bitwise_and(white_mask, field_region)

        # Step 4: Morphological cleanup to connect broken line segments
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        line_candidates = cv2.morphologyEx(line_candidates, cv2.MORPH_CLOSE, kernel_close)

        # Step 5: Canny edge detection
        edges = cv2.Canny(line_candidates, self.canny_low, self.canny_high)

        # Step 6: Probabilistic Hough Line Transform
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        lines = []
        horizontal_lines = []
        vertical_lines = []
        diagonal_lines = []

        if raw_lines is not None:
            for seg in raw_lines:
                coords = seg.ravel()
                if len(coords) != 4:
                    continue
                x1, y1, x2, y2 = coords
                lines.append((x1, y1, x2, y2))

                # Classify by angle
                angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if angle < 15:
                    horizontal_lines.append((x1, y1, x2, y2, length))
                elif angle > 75:
                    vertical_lines.append((x1, y1, x2, y2, length))
                else:
                    diagonal_lines.append((x1, y1, x2, y2, length))

        # Sort by line length (longest first — most reliable)
        horizontal_lines.sort(key=lambda l: l[4], reverse=True)
        vertical_lines.sort(key=lambda l: l[4], reverse=True)

        # Step 7: Circle detection (center circle, penalty arcs)
        blurred = cv2.GaussianBlur(line_candidates, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_dp,
            minDist=self.circle_min_dist,
            param1=self.circle_param1,
            param2=self.circle_param2,
            minRadius=self.circle_min_radius,
            maxRadius=self.circle_max_radius
        )

        detected_circles = []
        if circles is not None:
            for c in np.uint16(np.around(circles[0])):
                detected_circles.append((int(c[0]), int(c[1]), int(c[2])))

        return {
            'lines': lines,
            'horizontal_lines': [(l[0], l[1], l[2], l[3]) for l in horizontal_lines],
            'vertical_lines': [(l[0], l[1], l[2], l[3]) for l in vertical_lines],
            'diagonal_lines': [(l[0], l[1], l[2], l[3]) for l in diagonal_lines],
            'circles': detected_circles,
            'field_mask': field_region,
            'line_mask': line_candidates,
            'n_lines': len(lines),
            'n_circles': len(detected_circles)
        }

    def detect_field_lines_batch(
        self,
        frames: List[np.ndarray],
        sample_every_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Runs field line detection over a batch of frames.
        Samples every N frames for efficiency and reuses results for skipped frames.

        Args:
            frames: List of BGR video frames
            sample_every_n: Process every N-th frame

        Returns:
            List of field line dicts, one per frame.
        """
        results = []
        last_result = self._empty_result()

        for idx, frame in enumerate(frames):
            if idx % sample_every_n == 0:
                try:
                    last_result = self.detect_field_lines(frame)
                except Exception:
                    pass
            results.append(last_result)

        n_unique = sum(1 for i in range(0, len(frames), sample_every_n))
        total_lines = sum(r['n_lines'] for r in results[::sample_every_n] if r)
        avg_lines = total_lines / max(n_unique, 1)
        print(f"[FieldDetector] Processed {n_unique}/{len(frames)} frames. Avg lines per frame: {avg_lines:.1f}")
        return results

    def classify_pitch_regions(
        self,
        field_data: Dict[str, Any],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Classifies detected lines into semantic pitch regions.

        Uses heuristics based on line position and orientation to identify:
        - Sidelines (long horizontal lines near top/bottom)
        - Goal lines (short vertical lines at left/right edges)
        - Penalty box edges (horizontal lines near the goals)
        - Center line (vertical line near center)

        Returns:
            Dict with classified pitch elements.
        """
        center_x = frame_width / 2
        center_y = frame_height / 2

        sideline_candidates = []
        goal_line_candidates = []
        center_line_candidates = []
        penalty_box_candidates = []

        for line in field_data.get('horizontal_lines', []):
            x1, y1, x2, y2 = line
            mid_y = (y1 + y2) / 2
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if length > frame_width * 0.4:
                # Long horizontal line — likely sideline
                if mid_y < center_y:
                    sideline_candidates.append(('top_sideline', line))
                else:
                    sideline_candidates.append(('bottom_sideline', line))
            elif length > frame_width * 0.1:
                # Shorter horizontal line — penalty box edge
                penalty_box_candidates.append(line)

        for line in field_data.get('vertical_lines', []):
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if abs(mid_x - center_x) < frame_width * 0.1 and length > frame_height * 0.3:
                center_line_candidates.append(line)
            elif mid_x < frame_width * 0.15 or mid_x > frame_width * 0.85:
                goal_line_candidates.append(line)

        # Center circle detection
        center_circle = None
        for cx, cy, r in field_data.get('circles', []):
            if abs(cx - center_x) < frame_width * 0.15 and abs(cy - center_y) < frame_height * 0.25:
                center_circle = (cx, cy, r)
                break

        return {
            'sidelines': sideline_candidates,
            'goal_lines': goal_line_candidates,
            'center_line': center_line_candidates[0] if center_line_candidates else None,
            'center_circle': center_circle,
            'penalty_boxes': penalty_box_candidates
        }

    def get_dynamic_homography_anchors(
        self,
        classified_regions: Dict[str, Any],
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates image↔pitch point correspondences from detected field lines
        for dynamic homography refinement.

        Returns:
            Tuple of (image_points, pitch_points) as np.float32 arrays,
            or None if insufficient anchors detected.
        """
        image_pts = []
        pitch_pts = []

        # Center circle → pitch center (52.5m, 34m)
        if classified_regions.get('center_circle') is not None:
            cx, cy, r = classified_regions['center_circle']
            image_pts.append([cx, cy])
            pitch_pts.append([pitch_length / 2, pitch_width / 2])

        # Center line midpoint
        if classified_regions.get('center_line') is not None:
            x1, y1, x2, y2 = classified_regions['center_line']
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            image_pts.append([mid_x, mid_y])
            pitch_pts.append([pitch_length / 2, pitch_width / 2])

        # Sideline intersections
        for label, line in classified_regions.get('sidelines', []):
            x1, y1, x2, y2 = line
            if label == 'top_sideline':
                image_pts.append([x1, y1])
                pitch_pts.append([0.0, 0.0])
                image_pts.append([x2, y2])
                pitch_pts.append([pitch_length, 0.0])
            elif label == 'bottom_sideline':
                image_pts.append([x1, y1])
                pitch_pts.append([0.0, pitch_width])
                image_pts.append([x2, y2])
                pitch_pts.append([pitch_length, pitch_width])

        if len(image_pts) < 4:
            return None

        return (
            np.array(image_pts[:8], dtype=np.float32),
            np.array(pitch_pts[:8], dtype=np.float32)
        )

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Returns an empty detection result."""
        return {
            'lines': [],
            'horizontal_lines': [],
            'vertical_lines': [],
            'diagonal_lines': [],
            'circles': [],
            'field_mask': None,
            'line_mask': None,
            'n_lines': 0,
            'n_circles': 0
        }
