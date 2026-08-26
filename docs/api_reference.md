# Football Match Video Analysis System — API Reference

## Module Index

### 1. `src.config`
- `load_config(config_path="config.yaml") -> ConfigLoader`: Loads and returns the singleton configuration instance.
- `ConfigLoader.get(key_path, default=None)`: Retrieves configuration values using dot notation.

### 2. `src.video_io`
- `VideoIO(video_path)`: Video reader class.
- `VideoIO.read_frames(target_size=None)`: Reads all video frames into memory as a list of NumPy BGR images.
- `VideoIO.save_video(output_path, frames, fps=25.0)`: Writes a list of annotated frames to an MP4 file.

### 3. `src.scene_filter`
- `SceneFilter(green_ratio_threshold=0.35)`: Evaluates green grass proportion in HSV color space.
- `SceneFilter.is_action_frame(frame) -> bool`: Returns `True` if green pitch ratio exceeds the threshold.

### 4. `src.detector`
- `ObjectDetector(model_path, fallback_model, conf_thresh)`: Wrapper around Ultralytics YOLO.
- `ObjectDetector.detect_frame(frame) -> List[Dict]`: Returns detection list `[{'bbox': [...], 'class_id': int, 'conf': float}]`.

### 5. `src.tracker`
- `MultiObjectTracker(stub_path, track_high_thresh, track_buffer)`: ByteTrack multi-object tracking implementation.
- `MultiObjectTracker.track_frames(frames, detections_per_frame, read_from_stub=True)`: Returns persistent track ID assignments across frames.

### 6. `src.team_assigner`
- `TeamAssigner(n_clusters=2, mask_grass=True)`: Torso green-masked K-Means clustering engine.
- `TeamAssigner.fit_team_colors(frames, tracks_per_frame)`: Fits K-Means model on non-grass upper torso pixels.
- `TeamAssigner.get_player_team(frame, bbox, player_id) -> int`: Returns `0` (Team A) or `1` (Team B).

### 7. `src.camera_movement`
- `CameraMovementEstimator(stub_path)`: Lucas-Kanade Optical Flow camera movement estimator.
- `CameraMovementEstimator.get_camera_movement(frames, read_from_stub=True)`: Returns `(dx, dy)` camera shift per frame.

### 8. `src.homography`
- `HomographyTransformer(pitch_length=105.0, pitch_width=68.0)`: Homography transformation matrix engine.
- `HomographyTransformer.transform_bbox_bottom(bbox) -> Tuple[float, float]`: Transforms pixel bounding box feet position to metric pitch meters $(X_m, Y_m)$.

### 9. `src.analytics_engine`
- `AnalyticsEngine(fps=25.0, possession_proximity_thresh=2.0)`: Calculates distance, speed ($\text{km/h}$), ball possession percentage, and smoothed trajectories.

### 10. `src.event_detector`
- `EventDetector(fps=25.0)`: Rule engine detecting fouls and yellow card candidates.

### 11. `src.visualizer`
- `Visualizer(pitch_length=105.0, pitch_width=68.0)`: Overlays bboxes, speed tags, team badges, and 2D tactical pitch radar.
- `Visualizer.generate_heatmap_image(player_stats, output_path)`: Saves 2D density heatmap graphic.

### 12. `src.report_generator`
- `ReportGenerator(output_dir="reports")`: Exports `stats_player.csv`, `events.csv`, and interactive HTML dashboard `dashboard.html`.
