# Football Match Video Analysis System — System Architecture

## Overview
The **Football Match Video Analysis System** is an end-to-end computer vision and sports analytics platform designed to convert raw broadcast match footage into structured player/team metrics, event logs, and annotated video streams without manual human labeling.

---

## 10-Stage Pipeline Flow Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FOOTBALL ANALYSIS PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │   VIDEO      │    │   SCENE      │    │   OBJECT     │                 │
│  │   INGESTION  │───▶│   FILTER     │───▶│   DETECTION  │                 │
│  │  (video_io)  │    │(scene_filter)│    │   (YOLO)     │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │   TEAM       │    │   MULTI-     │    │   TRACKING   │                 │
│  │   ASSIGNMENT │◀───│   OBJECT     │◀───│   (ByteTrack)│                 │
│  │   (K-Means)  │    │   TRACKING   │    │   (tracker)  │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │   CAMERA     │    │   PERSPECTIVE│    │   ANALYTICS  │                 │
│  │   MOTION     │───▶│   TRANSFORM  │───▶│   ENGINE     │                 │
│  │(camera_move) │    │ (Homography) │    │  (analytics) │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                          │
│                                                  ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │   EVENT      │    │   OUTPUT     │    │   REPORT     │                 │
│  │   DETECTION  │───▶│   VIDEO      │───▶│   GENERATION │                 │
│  │(event_detect)│    │ (Visualizer) │    │ (HTML/CSV)   │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Subsystems & Design Choices

1. **Scene Filter (`scene_filter.py`)**:
   - Converts BGR frames to HSV space and calculates the ratio of pitch green grass pixels ($H \in [35,85]$).
   - Automatically drops non-action replay cuts, spectator close-ups, and studio camera angles.

2. **Object Detector (`detector.py`)**:
   - Uses YOLO (fine-tuned `best.pt` or fallback `yolov8x.pt`) to detect player, goalkeeper, referee, and ball bounding boxes.

3. **Multi-Object Tracker (`tracker.py`)**:
   - Uses ByteTrack logic with IoU association and track buffers.
   - Pickles track outputs into `stubs/tracks.pkl` for fast downstream recalculations.

4. **Team Assigner (`team_assigner.py`)**:
   - Isolates the top 40% torso region of player bounding boxes.
   - Masks out pitch green grass background pixels.
   - Fits a 2-cluster K-Means model on non-grass RGB/LAB color features to partition players into Team A (Red) and Team B (Blue).

5. **Camera Motion Estimator (`camera_movement.py`)**:
   - Tracks background pitch feature keypoints across frames using Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`).
   - Computes $(dx, dy)$ frame shift to isolate true player motion on the pitch from broadcast camera pan/zoom.

6. **Homography Transformer (`homography.py`)**:
   - Maps frame pixel coordinates $(x, y)$ to real-world metric pitch coordinates $(X_m, Y_m)$ using standard 4-point homography transform matrix $H$.
   - Metric Pitch Standard: $105.0\text{ meters} \times 68.0\text{ meters}$.

7. **Analytics Engine (`analytics_engine.py`)**:
   - Trajectory smoothing via rolling moving average.
   - Derivatives for speed ($\text{m/s}$ & $\text{km/h}$) and cumulative distance ($\text{m}$).
   - Ball possession calculation based on spatial proximity mapping.

8. **Event Detector (`event_detector.py`)**:
   - Evaluates player-player proximity ($<1.2\text{m}$), ball velocity drops, and referee proximity ($<1.5\text{m}$) to flag potential fouls and yellow card candidates with timestamped confidence scores.

9. **Visualizer & Report Generator (`visualizer.py`, `report_generator.py`)**:
   - Renders annotated output video (`data/output_videos/match_01_annotated.mp4`) with player badges, speed meters, and a top-down 2D tactical pitch radar.
   - Exports CSV datasets (`stats_player.csv`, `events.csv`), spatial position heatmaps (`heatmap.png`), and an interactive HTML report dashboard (`dashboard.html`).
