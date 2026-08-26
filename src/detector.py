import os
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    
    # Fix for PyTorch 2.6+ weights_only=True blocking custom YOLO model loads
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
        
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

class ObjectDetector:
    """
    Stage 3: Wraps YOLO inference to detect players, goalkeepers, referees, and the ball.
    Outputs detections per frame as dicts containing bounding boxes, class IDs, and confidence scores.
    """
    def __init__(
        self,
        model_path: str = "models/weights/best.pt",
        fallback_model: str = "yolov8x.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "auto"
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None

        if not ULTRALYTICS_AVAILABLE:
            print("Warning: ultralytics package is not installed. Detector will run in mock mode unless installed.")
            return

        target_model = model_path if os.path.exists(model_path) else fallback_model
        print(f"Loading YOLO detector with weights: {target_model}")
        try:
            self.model = YOLO(target_model)
        except Exception as e:
            print(f"Error loading YOLO model ({e}). Initializing fallback YOLO model.")
            self.model = YOLO(fallback_model)

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single image frame.
        Returns a list of dicts: [{'bbox': [x1, y1, x2, y2], 'class_id': int, 'conf': float}]
        """
        if self.model is None:
            # Fallback mock detection for environment without model weights loaded
            return self._generate_mock_detections(frame)

        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Map COCO classes to football classes if standard YOLO model is used
                mapped_cls_id = self._map_class_id(cls_id)
                if mapped_cls_id is not None:
                    detections.append({
                        'bbox': xyxy,
                        'class_id': mapped_cls_id,
                        'conf': conf
                    })
        return detections

    def detect_frames(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Runs detection on a batch or sequence of frames.
        """
        return [self.detect_frame(frame) for frame in frames]

    def _map_class_id(self, cls_id: int) -> Optional[int]:
        """
        Maps model class IDs to football pipeline IDs dynamically based on model class names.
        Pipeline IDs: 0: player, 1: goalkeeper, 2: referee, 3: ball
        """
        if self.model is None or not hasattr(self.model, 'names'):
            return 0
            
        class_name = self.model.names.get(cls_id, "").lower()
        
        if class_name in ("player", "person"):
            return 0
        elif class_name == "goalkeeper":
            return 1
        elif class_name == "referee":
            return 2
        elif class_name in ("ball", "sports ball"):
            return 3
            
        return 0 # Default to player for unknown human-like detections

    def _generate_mock_detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Mock detection generator used during testing or headless validation.
        """
        h, w = frame.shape[:2]
        return [
            {'bbox': [w * 0.3, h * 0.4, w * 0.35, h * 0.6], 'class_id': 0, 'conf': 0.92},
            {'bbox': [w * 0.6, h * 0.4, w * 0.65, h * 0.6], 'class_id': 0, 'conf': 0.88},
            {'bbox': [w * 0.1, h * 0.4, w * 0.15, h * 0.6], 'class_id': 1, 'conf': 0.95},
            {'bbox': [w * 0.5, h * 0.3, w * 0.54, h * 0.5], 'class_id': 2, 'conf': 0.85},
            {'bbox': [w * 0.48, h * 0.55, w * 0.50, h * 0.57], 'class_id': 3, 'conf': 0.90},
        ]
