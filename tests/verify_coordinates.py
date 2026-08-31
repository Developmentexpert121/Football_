# verify_coordinates.py
# Standalone coordinate diagnostic: validates all goal polygons, net ROIs,
# and homography reference points, then saves annotated frame image.
#
# Usage:  python tests/verify_coordinates.py
# Output: reports/coordinate_verification.png + console diagnostic table

import cv2
import numpy as np
import yaml
import os
import sys

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("ERROR: config.yaml not found. Run from project root.")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    video_path = cfg["paths"]["input_video"]
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Read frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: Could not read frame 100")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Extract config values
    ref_img = np.float32(cfg["pitch"]["reference_points_image"])
    ref_pitch = np.float32(cfg["pitch"]["reference_points_pitch"])
    left_poly = cfg["event_detector"]["left_goal_polygon"]
    right_poly = cfg["event_detector"]["right_goal_polygon"]
    left_net = cfg["event_detector"]["left_net_roi"]
    right_net = cfg["event_detector"]["right_net_roi"]

    # Compute homography
    H, _ = cv2.findHomography(ref_img, ref_pitch)
    H_inv = np.linalg.inv(H)

    print("\n" + "=" * 72)
    print("  COORDINATE VERIFICATION DIAGNOSTIC")
    print("=" * 72)

    # 1. Reference points check
    print("\n--- Homography Reference Points ---")
    labels = ["TL", "TR", "BR", "BL"]
    for i, (img_pt, pitch_pt) in enumerate(zip(ref_img, ref_pitch)):
        mapped = cv2.perspectiveTransform(np.float32([[img_pt]]), H)
        mx, my = mapped[0][0]
        ex, ey = pitch_pt
        err = np.sqrt((mx - ex)**2 + (my - ey)**2)
        status = "PASS" if err < 0.5 else "FAIL"
        print(f"  {labels[i]}: pixel [{int(img_pt[0])}, {int(img_pt[1])}] "
              f"-> [{mx:.2f}m, {my:.2f}m] "
              f"(expected [{ex:.1f}m, {ey:.1f}m]) "
              f"err={err:.3f}m [{status}]")

    # 2. Left goal polygon check
    print("\n--- Left Goal Polygon ---")
    for i, pt in enumerate(left_poly):
        mapped = cv2.perspectiveTransform(np.float32([[[pt[0], pt[1]]]]), H)
        mx, my = mapped[0][0]
        on_goal_line = abs(mx) < 2.0
        in_goal_y = 28.0 <= my <= 40.0
        status = "PASS" if on_goal_line else "CHECK"
        print(f"  P{i+1}: pixel [{pt[0]}, {pt[1]}] "
              f"-> [{mx:.2f}m, {my:.2f}m] "
              f"goal_line={on_goal_line} goal_y={in_goal_y} [{status}]")

    # 3. Right goal polygon check
    print("\n--- Right Goal Polygon ---")
    for i, pt in enumerate(right_poly):
        mapped = cv2.perspectiveTransform(np.float32([[[pt[0], pt[1]]]]), H)
        mx, my = mapped[0][0]
        on_goal_line = abs(mx - 105.0) < 2.0
        in_goal_y = 28.0 <= my <= 40.0
        status = "PASS" if on_goal_line else "CHECK"
        print(f"  P{i+1}: pixel [{pt[0]}, {pt[1]}] "
              f"-> [{mx:.2f}m, {my:.2f}m] "
              f"goal_line={on_goal_line} goal_y={in_goal_y} [{status}]")

    # 4. Net ROI check
    print("\n--- Net ROI Bounds ---")
    for name, roi in [("Left Net", left_net), ("Right Net", right_net)]:
        corners = [
            [roi[0], roi[1]], [roi[2], roi[1]],
            [roi[0], roi[3]], [roi[2], roi[3]]
        ]
        print(f"  {name} ROI: {roi}")
        for j, c in enumerate(corners):
            mapped = cv2.perspectiveTransform(np.float32([[[c[0], c[1]]]]), H)
            mx, my = mapped[0][0]
            print(f"    corner{j+1} [{c[0]},{c[1]}] -> [{mx:.2f}m, {my:.2f}m]")

    print("\n" + "=" * 72)

    # ── Draw everything on the frame ──────────────────────────
    annotated = frame.copy()

    # Reference points
    ref_colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    for i, pt in enumerate(ref_img):
        px, py = int(pt[0]), int(pt[1])
        cv2.drawMarker(annotated, (px, py), ref_colors[i], cv2.MARKER_CROSS, 20, 2)
        cv2.putText(annotated, f"REF-{labels[i]}", (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ref_colors[i], 2)

    # Left goal polygon (green)
    pts_l = np.array(left_poly, dtype=np.int32)
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [pts_l], (0, 200, 0))
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    cv2.polylines(annotated, [pts_l], True, (0, 255, 0), 2)
    for i, (px, py) in enumerate(left_poly):
        cv2.circle(annotated, (int(px), int(py)), 6, (0, 0, 255), -1)
        cv2.putText(annotated, f"LG-P{i+1}", (int(px)+8, int(py)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Right goal polygon (blue)
    pts_r = np.array(right_poly, dtype=np.int32)
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [pts_r], (200, 100, 0))
    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
    cv2.polylines(annotated, [pts_r], True, (255, 165, 0), 2)
    for i, (px, py) in enumerate(right_poly):
        cv2.circle(annotated, (int(px), int(py)), 6, (0, 0, 255), -1)
        cv2.putText(annotated, f"RG-P{i+1}", (int(px)+8, int(py)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 2)

    # Net ROIs (dashed rectangles)
    cv2.rectangle(annotated, (left_net[0], left_net[1]),
                  (left_net[2], left_net[3]), (128, 255, 128), 2)
    cv2.putText(annotated, "L-NET-ROI", (left_net[0], left_net[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 2)

    cv2.rectangle(annotated, (right_net[0], right_net[1]),
                  (right_net[2], right_net[3]), (128, 128, 255), 2)
    cv2.putText(annotated, "R-NET-ROI", (right_net[0], right_net[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 255), 2)

    # Title
    cv2.putText(annotated, "COORDINATE VERIFICATION", (w//2 - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/coordinate_verification.png"
    cv2.imwrite(out_path, annotated)
    print(f"\nVerification image saved to: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()