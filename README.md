# ⚽ Football Match Video Analysis System

An automated, end-to-end computer vision and sports analytics pipeline that converts raw broadcast football match video into player-level metrics (speed/distance), team possession analytics, heuristic event logs (fouls/cards), spatial heatmaps, and an annotated output video with tactical mini-map radar—all with **zero manual labeling**.

---

## 🌟 Features & Pipeline Overview

1. **Video Ingestion**: Asynchronous frame extraction & video writing.
2. **Scene Filter**: HSV green grass ratio filter to automatically skip replays, spectator crowd shots, and close-up interviews.
3. **Object Detection**: Ultralytics YOLO (fine-tuned `best.pt` or fallback `yolov8x.pt`) detecting players, goalkeepers, referees, and the ball.
4. **Multi-Object Tracking**: ByteTrack persistent object tracking with disk-stub caching (`stubs/tracks.pkl`).
5. **Team Assignment**: Upper torso extraction (top 40% of bounding box) with HSV pitch green masking and 2-cluster K-Means color classification (Team A vs. Team B).
6. **Camera Motion Estimation**: Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) on pitch background keypoints to compensate for broadcast camera pan/zoom.
7. **Perspective Transformation (Homography)**: Converts image pixel coordinates $(x, y)$ to real-world pitch metric coordinates $(X_m, Y_m)$ ($105.0\text{m} \times 68.0\text{m}$).
8. **Analytics Engine**:
   - Trajectory smoothing via rolling moving average.
   - Player speed ($\text{m/s}$ & $\text{km/h}$) and cumulative distance covered ($\text{m}$).
   - Ball possession percentage breakdown per team based on spatial proximity mapping.
   - Spatial position distribution heatmaps.
9. **Event Detection**: Rule engine flagging potential foul candidates (player collision + ball deceleration) and yellow card incidents (referee proximity + player stoppage).
10. **Deliverables & Reporting**:
    - Annotated video (`data/output_videos/match_01_annotated.mp4`) with bounding box badges, speed indicators, and top-down tactical pitch radar mini-map.
    - Stats CSV exports (`reports/stats_player.csv`, `reports/events.csv`).
    - Spatial position heatmap graphic (`reports/heatmap.png`).
    - Interactive standalone HTML dashboard (`reports/dashboard.html`).

---

## 📁 Directory Structure

```text
football_analytics_project/
│
├── README.md                         # Complete project overview & execution guide
├── requirements.txt                  # Python dependencies
├── config.yaml                       # SINGLE SOURCE OF TRUTH: tunable parameters
├── main.py                           # 10-stage pipeline orchestrator
├── run.py                            # CLI entry point (with synthetic demo generator)
│
├── src/                              # Core pipeline modules
│   ├── __init__.py
│   ├── config.py                     # Configuration loader
│   ├── video_io.py                   # Video frame reading/writing
│   ├── scene_filter.py               # Green grass scene filter
│   ├── detector.py                   # YOLO detector wrapper
│   ├── tracker.py                    # ByteTrack tracker + stub caching
│   ├── team_assigner.py              # Torso K-Means jersey color classifier
│   ├── camera_movement.py            # Optical flow camera estimator
│   ├── homography.py                 # Pixel -> meters transform
│   ├── jersey_ocr.py                 # Priority VLM (SmolVLM2) jersey number extraction engine
│   ├── analytics_engine.py           # Speed, distance, possession, heatmaps
│   ├── event_detector.py             # Foul & card candidate rule engine
│   ├── visualizer.py                 # Annotations, heatmaps & mini-map overlay
│   └── report_generator.py          # CSV exporter & interactive HTML dashboard
│
├── models/
│   └── weights/                      # Trained model weights (best.pt, yolov8x.pt)
│
├── data/
│   ├── input_videos/                 # Raw match video inputs
│   └── output_videos/                # Annotated analysis videos
│
├── stubs/                            # Pickled intermediate calculation cache
│   ├── detections.pkl
│   ├── tracks.pkl
│   └── camera_movement.pkl
│
├── reports/                          # Generated analytics reports
│   ├── stats_player.csv
│   ├── events.csv
│   ├── heatmap.png
│   └── dashboard.html
│
└── tests/                            # Unit test suite
    ├── test_scene_filter.py
    ├── test_detector.py
    ├── test_tracker.py
    ├── test_team_assigner.py
    ├── test_homography.py
    └── test_analytics.py
```

---

## 🚀 Quick Start & Usage

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Synthetic Demo (Instant Testing)
To verify the entire 10-stage pipeline immediately without downloading large match videos:
```bash
python run.py --demo
```
This automatically creates a synthetic 1280x720 football match video clip, executes all 10 pipeline stages, and generates:
- `data/output_videos/match_01_annotated.mp4`
- `reports/dashboard.html`
- `reports/heatmap.png`
- `reports/stats_player.csv`
- `reports/events.csv`

### 3. Run on Custom Match Video
Place your match `.mp4` file into `data/input_videos/match_01.mp4` and run:
```bash
python main.py --input data/input_videos/match_01.mp4 --output data/output_videos/match_01_annotated.mp4
```

To bypass cached stub data and recompute tracking from scratch:
```bash
python main.py --no-stubs
```

---

## 🧪 Running Unit Tests

Run the Pytest suite across all core modules:
```bash
python -m pytest tests/
```

---

## ⚙️ Configuration

All pipeline thresholds, FPS, pitch dimensions ($105\text{m} \times 68\text{m}$), YOLO weights, and visual colors are managed centrally in `config.yaml`.
