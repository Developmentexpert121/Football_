import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np

class MultiObjectTracker:
    """
    Stage 4: Multi-Object Tracking using ByteTrack logic.
    Maintains persistent IDs across frames and handles disk stub caching (stubs/tracks.pkl).
    """
    def __init__(
        self,
        stub_path: str = "stubs/tracks.pkl",
        track_high_thresh: float = 0.5,
        track_buffer: int = 30
    ):
        self.stub_path = stub_path
        self.track_high_thresh = track_high_thresh
        self.track_buffer = track_buffer
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1

    def track_frames(
        self,
        frames: List[np.ndarray],
        detections_per_frame: List[List[Dict[str, Any]]],
        read_from_stub: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Processes detections per frame and assigns track IDs.
        If read_from_stub=True and stubs/tracks.pkl exists, loads from stub cache instantly.
        """
        if read_from_stub and os.path.exists(self.stub_path):
            try:
                with open(self.stub_path, "rb") as f:
                    stub_data = pickle.load(f)
                    if isinstance(stub_data, list) and len(stub_data) == len(frames):
                        print(f"Loading cached tracking results from stub: {self.stub_path}")
                        return stub_data
                    else:
                        print(f"Stub tracking frame count ({len(stub_data) if isinstance(stub_data, list) else 'invalid'}) does not match input frame count ({len(frames)}). Recomputing tracking...")
            except Exception as e:
                print(f"Error reading tracking stub cache ({e}). Recomputing tracking...")

        tracked_output = []
        self.active_tracks = {}
        self.next_track_id = 1

        for frame_idx, detections in enumerate(detections_per_frame):
            frame_tracks = self._update_tracks_for_frame(detections)
            tracked_output.append(frame_tracks)

        # Save to stub directory
        os.makedirs(os.path.dirname(self.stub_path), exist_ok=True)
        with open(self.stub_path, "wb") as f:
            pickle.dump(tracked_output, f)
        print(f"Saved tracking results to stub cache: {self.stub_path}")

        return tracked_output

    def _update_tracks_for_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Associates detections to active tracks using IoU matching.
        """
        frame_tracks = []
        unmatched_detections = list(detections)

        # Simple greedy IoU matching between existing tracks and new detections
        matched_track_ids = set()

        for track_id, track_data in list(self.active_tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            prev_bbox = track_data['bbox']

            for det_idx, det in enumerate(unmatched_detections):
                iou = self._compute_iou(prev_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= 0.3 and best_det_idx >= 0:
                det = unmatched_detections.pop(best_det_idx)
                self.active_tracks[track_id] = {
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'conf': det['conf'],
                    'lost_count': 0
                }
                matched_track_ids.add(track_id)
                frame_tracks.append({
                    'track_id': track_id,
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'conf': det['conf']
                })
            else:
                self.active_tracks[track_id]['lost_count'] += 1
                if self.active_tracks[track_id]['lost_count'] > self.track_buffer:
                    del self.active_tracks[track_id]

        # Assign new track IDs to unmatched detections
        for det in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[track_id] = {
                'bbox': det['bbox'],
                'class_id': det['class_id'],
                'conf': det['conf'],
                'lost_count': 0
            }
            frame_tracks.append({
                'track_id': track_id,
                'bbox': det['bbox'],
                'class_id': det['class_id'],
                'conf': det['conf']
            })

        return frame_tracks

    @staticmethod
    def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0
