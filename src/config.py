import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    """
    Loads and manages project configuration parameters from config.yaml.
    Provides fallback defaults if config fields are omitted.
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_default_config()

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves nested config key using dot notation (e.g. 'detector.confidence_threshold').
        """
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._config

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "paths": {
                "input_video": "data/input_videos/match_01.mp4",
                "output_video": "data/output_videos/match_01_annotated.mp4",
                "stubs_dir": "stubs",
                "reports_dir": "reports"
            },
            "video": {"target_fps": 25, "process_width": 1280, "process_height": 720},
            "scene_filter": {"enabled": True, "green_ratio_threshold": 0.35},
            "detector": {"confidence_threshold": 0.25, "iou_threshold": 0.45},
            "tracker": {"track_high_thresh": 0.5, "track_buffer": 30},
            "team_assigner": {"n_clusters": 2, "mask_grass": True},
            "pitch": {"length_meters": 105.0, "width_meters": 68.0},
            "analytics": {"fps": 25, "possession_proximity_meters": 2.0}
        }

def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    return ConfigLoader(config_path)
