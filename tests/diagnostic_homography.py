# diagnostic_homography.py
# Diagnostic test for homography calibration
import cv2
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

image_pts = np.float32(cfg["pitch"]["reference_points_image"])
pitch_pts = np.float32(cfg["pitch"]["reference_points_pitch"])

H, _ = cv2.findHomography(image_pts, pitch_pts)

left_poly = cfg["event_detector"]["left_goal_polygon"]
left_poly_pixels = np.float32([[[left_poly[0][0], left_poly[0][1]]], [[left_poly[3][0], left_poly[3][1]]]])

result = cv2.perspectiveTransform(left_poly_pixels, H)
print("=" * 65)
print("  HOMOGRAPHY DIAGNOSTIC CHECK")
print("=" * 65)
print("Left goal pixel -> ground metres:")
print(f"  P1 (left post base):  X={result[0][0][0]:.2f}m, Y={result[0][0][1]:.2f}m")
print(f"  P4 (crossbar top):   X={result[1][0][0]:.2f}m, Y={result[1][0][1]:.2f}m")
print()
print("Expected: X ~= 0.00m, Y between 30.34m - 37.66m")
print("Status: CALIBRATION PERFECT" if abs(result[0][0][0]) < 0.1 and 30 <= result[0][0][1] <= 38 else "Status: RECALIBRATION REQUIRED")
print("=" * 65)