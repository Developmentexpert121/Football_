"""
SoccerNet Goal Detector — Standalone Module (goal-improve branch)
================================================================
Integrates the pretrained action_sampling_weights_002 model from:
  https://github.com/lRomul/ball-action-spotting

Usage (on Google Colab after running colab_setup_goal_improve.ipynb):
    from src.soccernet_goal_detector import SoccerNetGoalDetector

    detector = SoccerNetGoalDetector(
        checkpoint_path='/content/weights/action_sampling_weights_002/model.pt',
        repo_path='/content/ball-action-spotting',
        device='cuda'
    )
    goals = detector.detect('/content/match.mp4')
    # Returns: [(8.4, 0.91), (23.7, 0.87)]

NOTE: This module is 100% standalone. It does NOT modify main.py or any
      existing pipeline code. It lives only on the goal-improve branch.
"""

import os
import sys
import time
import importlib
import importlib.util
import numpy as np
import cv2
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# SoccerNet 15-class action spotting index mapping
# Source: configs/action/action_sampling_weights_002.py
SOCCERNET_CLASSES = [
    "Penalty",            # 0
    "Kick-off",           # 1
    "Goal",               # 2  <- WE ONLY USE THIS
    "Substitution",       # 3
    "Offside",            # 4
    "Shots on target",    # 5
    "Shots off target",   # 6
    "Clearance",          # 7
    "Ball out of play",   # 8
    "Throw-in",           # 9
    "Foul",               # 10
    "Indirect free-kick", # 11
    "Direct free-kick",   # 12
    "Corner",             # 13
    "Card",               # 14
]

GOAL_CLASS_INDEX = 2        # Index of Goal in SOCCERNET_CLASSES

# Model input parameters from README + action_sampling_weights_002 config
FRAME_WIDTH = 1280
FRAME_HEIGHT = 736
SEQUENCE_LENGTH = 15        # 15 consecutive 1280x736 grayscale frames
EFFECTIVE_FPS = 12.5         # Skip every 2nd frame from 25 FPS source


class SoccerNetGoalDetector:
    """
    Standalone goal detector using the SoccerNet ball-action-spotting
    pretrained model (action_sampling_weights_002 - 15-class action spotting).
    """

    def __init__(
        self,
        checkpoint_path: str,
        repo_path: str = "/content/ball-action-spotting",
        device: str = "cuda",
        gaussian_sigma: float = 3.0,
        peak_min_height: float = 0.2,
        peak_min_distance_frames: int = 15,
        replay_merge_seconds: float = 45.0,
        batch_size: int = 4,
        **kwargs
    ):
        self.checkpoint_path = checkpoint_path
        self.repo_path = repo_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gaussian_sigma = gaussian_sigma
        self.peak_min_height = peak_min_height
        self.peak_min_distance_frames = peak_min_distance_frames
        self.replay_merge_seconds = replay_merge_seconds
        self.batch_size = batch_size

        print(f"[SoccerNetGoalDetector v2.5] Device: {self.device}")
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Download from: https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz"
            )

        print(f"[SoccerNetGoalDetector v3.0] Loading checkpoint: {self.checkpoint_path}")

        # 1. Resolve ball-action-spotting source directory and Football_ source directory
        ball_src = os.path.join(self.repo_path, "src")
        football_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

        # 2. Add repo and repo_src at top of sys.path
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        if ball_src not in sys.path:
            sys.path.insert(0, ball_src)

        # 3. Create synthetic namespace module for 'src' containing BOTH ball_src and football_src
        import types
        src_ns = types.ModuleType("src")
        src_ns.__path__ = [ball_src, football_src]
        src_ns.__package__ = "src"

        # Remove stale cached src.* submodules from sys.modules
        for k in list(sys.modules.keys()):
            if k.startswith("src."):
                del sys.modules[k]

        sys.modules['src'] = src_ns

        import argus

        # 4. Explicitly load ball-action-spotting submodules into 'src'
        try:
            import src.models.multidim_stacker
            import src.losses
            import src.mixup
            import src.argus_models
            from src.argus_models import BallActionModel

            model = argus.load_model(self.checkpoint_path, device=str(self.device))
            print(f"[SoccerNetGoalDetector] ✅ Model loaded via argus.load_model ({len(SOCCERNET_CLASSES)} classes)")
            return model
        except Exception as e_argus:
            print(f"[SoccerNetGoalDetector] argus.load_model notice: {e_argus}")
            try:
                from src.argus_models import BallActionModel
                chk = torch.load(self.checkpoint_path, map_location=self.device)
                params = chk['params'] if isinstance(chk, dict) and 'params' in chk else chk
                model = BallActionModel(params, device=str(self.device))
                if isinstance(chk, dict) and 'model_state_dict' in chk:
                    model.load_state_dict(chk['model_state_dict'])
                print(f"[SoccerNetGoalDetector] ✅ Model loaded via direct BallActionModel")
                return model
            except Exception as e_direct:
                raise RuntimeError(f"[SoccerNetGoalDetector] Failed to load checkpoint: {e_direct}")
        finally:
            if hasattr(sys.modules.get('src'), '__path__'):
                if football_src not in sys.modules['src'].__path__:
                    sys.modules['src'].__path__.append(football_src)

    def _preprocess_video(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_n = max(1, round(source_fps / EFFECTIVE_FPS))
        print(f"[SoccerNetGoalDetector] FPS: {source_fps:.1f} | Skip: {skip_n} -> {source_fps/skip_n:.1f} effective FPS")

        frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip_n == 0:
                resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            frame_idx += 1
        cap.release()
        print(f"[SoccerNetGoalDetector] Decoded {len(frames)} frames")
        return frames, source_fps, skip_n

    def _build_input_tensors(self, frames):
        """
        Build (15, 736, 1280) sequence tensors per frame.
        Batched into shape (B, 15, 736, 1280) as required by MultiDimStacker.forward_2d(x).
        """
        sequences = []
        num_frames = len(frames)
        norm_frames = [f.astype(np.float32) / 255.0 for f in frames]
        half_seq = SEQUENCE_LENGTH // 2  # 7

        for center in range(num_frames):
            indices = [min(max(center + offset, 0), num_frames - 1)
                       for offset in range(-half_seq, half_seq + 1)]
            # Stack 15 frames directly as (15, H, W)
            seq_frames = [norm_frames[idx] for idx in indices]
            sequence = np.stack(seq_frames, axis=0)  # (15, 736, 1280)
            sequences.append(torch.from_numpy(sequence))
        return sequences

    def _run_inference(self, sequences):
        all_scores = []
        n = len(sequences)
        print(f"[SoccerNetGoalDetector] Running inference on {n} frames...")
        t0 = time.time()

        if hasattr(self.model, 'predict'):
            with torch.no_grad():
                for start in range(0, n, self.batch_size):
                    batch = torch.stack(sequences[start: start + self.batch_size], dim=0).to(self.device)
                    res = self.model.predict(batch)
                    if isinstance(res, torch.Tensor):
                        probs = torch.sigmoid(res).cpu().numpy()
                    elif isinstance(res, np.ndarray):
                        if res.max() > 1.0 or res.min() < 0.0:
                            probs = 1.0 / (1.0 + np.exp(-res))
                        else:
                            probs = res
                    all_scores.append(probs)
                    if (start // self.batch_size) % 100 == 0:
                        print(f"  [{min(start+self.batch_size, n)}/{n}] {time.time()-t0:.1f}s")
        else:
            nn_module = getattr(self.model, 'nn_module', self.model)
            if hasattr(nn_module, 'eval'):
                nn_module.eval()
            with torch.no_grad():
                for start in range(0, n, self.batch_size):
                    batch = torch.stack(sequences[start: start + self.batch_size], dim=0).to(self.device)
                    logits = nn_module(batch)
                    if isinstance(logits, torch.Tensor):
                        probs = torch.sigmoid(logits).cpu().numpy()
                    elif isinstance(logits, np.ndarray):
                        probs = 1.0 / (1.0 + np.exp(-logits))
                    all_scores.append(probs)
                    if (start // self.batch_size) % 100 == 0:
                        print(f"  [{min(start+self.batch_size, n)}/{n}] {time.time()-t0:.1f}s")

        all_scores = np.concatenate(all_scores, axis=0)
        print(f"[SoccerNetGoalDetector] Inference done in {time.time()-t0:.1f}s")
        return all_scores

    def _smooth_and_find_peaks(self, goal_scores, effective_fps):
        smoothed = gaussian_filter1d(goal_scores, sigma=self.gaussian_sigma)
        peaks, _ = find_peaks(smoothed, height=self.peak_min_height, distance=self.peak_min_distance_frames)
        if len(peaks) == 0:
            return []

        candidates = sorted([(p / effective_fps, float(smoothed[p])) for p in peaks])
        print(f"[SoccerNetGoalDetector] Raw peaks: {len(candidates)}")

        # Replay deduplication
        deduplicated = []
        for ts, conf in candidates:
            if deduplicated and (ts - deduplicated[-1][0]) < self.replay_merge_seconds:
                if conf > deduplicated[-1][1]:
                    deduplicated[-1] = (ts, conf)
            else:
                deduplicated.append((ts, conf))

        print(f"[SoccerNetGoalDetector] After dedup: {len(deduplicated)} goal(s)")
        return deduplicated

    def detect(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect goals in any .mp4 file.

        Returns
        -------
        List[(timestamp_seconds, confidence)]
        Example: [(8.4, 0.91), (23.7, 0.87)]
        """
        print(f"\n{'='*60}")
        print(f"[SoccerNetGoalDetector] Detecting: {os.path.basename(video_path)}")
        print(f"{'='*60}")

        frames, source_fps, skip_n = self._preprocess_video(video_path)
        if len(frames) < SEQUENCE_LENGTH:
            print(f"Video too short ({len(frames)} frames). Need >= {SEQUENCE_LENGTH}.")
            return []

        sequences = self._build_input_tensors(frames)
        all_scores = self._run_inference(sequences)
        goal_scores = all_scores[:, GOAL_CLASS_INDEX]
        actual_fps = source_fps / skip_n
        goals = self._smooth_and_find_peaks(goal_scores, actual_fps)

        print(f"\n{'='*60}")
        if goals:
            print(f"[SoccerNetGoalDetector] GOALS DETECTED: {len(goals)}")
            for i, (ts, conf) in enumerate(goals, 1):
                m, s = int(ts // 60), int(ts % 60)
                print(f"  Goal #{i}: {m:02d}:{s:02d} ({ts:.1f}s) — confidence: {conf:.3f}")
        else:
            print("[SoccerNetGoalDetector] No goals detected.")
        print(f"{'='*60}\n")
        return goals
