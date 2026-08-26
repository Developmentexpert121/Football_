"""
Stage 8B: Priority Jersey Number OCR (VLM Engine)

Vision-Based Reading Like Human Eye using SmolVLM2, EasyOCR, or Gemini.
Tuned for broadcast match footage with multi-scale cropping and robust voting.
"""
import cv2
import re
import time
import os
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

# Graceful import handling
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class JerseyOCR:

    # COCO pose keypoint indices
    KP_LEFT_SHOULDER  = 5
    KP_RIGHT_SHOULDER = 6
    KP_LEFT_HIP       = 11
    KP_RIGHT_HIP      = 12

    # Thresholds — tuned for standard broadcast football
    KP_CONF_THRESHOLD  = 0.20
    OCR_CONF_THRESHOLD = 0.35   # Accessible threshold for small broadcast numbers
    MIN_BBOX_AREA      = 400    # px² — accommodates normal broadcast camera players
    MIN_VOTES          = 1      # Confident single-shot detection accepted
    MIN_VOTE_SHARE     = 0.35
    BACK_CROP_WEIGHT   = 1.5
    FRONT_CROP_WEIGHT  = 0.8
    CHAR_FIXES = str.maketrans("OoIlSsBbZz", "0011558822")

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.backend = "none"
        self._setup_backend()

    def _setup_backend(self):
        """
        Auto-selects best available backend:
        1. Gemini API  (no GPU needed, most accurate) - if GEMINI_API_KEY present
        2. SmolVLM2    (local VLM)
        3. EasyOCR     (local CPU/GPU fallback)
        """
        backend_cfg = self.config.get("jersey_ocr", {})
        backend = backend_cfg.get("backend", "auto")

        if (backend == "gemini" or backend == "auto") and os.environ.get("GEMINI_API_KEY") and GEMINI_AVAILABLE:
            self._init_gemini()
        elif (backend == "smolvlm2" or backend == "auto") and TRANSFORMERS_AVAILABLE:
            self._init_smolvlm2()
        elif EASYOCR_AVAILABLE:
            self._init_easyocr()
        else:
            logger.error("[JerseyOCR] No OCR backend available. Please install transformers or easyocr.")
            self.backend = "disabled"

    def _init_gemini(self):
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            self.vlm         = genai.GenerativeModel("gemini-2.0-flash")
            self.backend     = "gemini"
            self._call_count = 0
            self._last_reset = time.time()
            print("[JerseyOCR] Backend: Gemini 2.0 Flash")
        except Exception as e:
            logger.error(f"[JerseyOCR] Gemini init failed: {e}")
            if EASYOCR_AVAILABLE:
                self._init_easyocr()
            else:
                self.backend = "disabled"

    def _init_smolvlm2(self):
        model_id = self.config.get("jersey_ocr", {}).get(
            "smolvlm2_model",
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.environ.get("HF_TOKEN")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                token=hf_token
            ).eval()
            self.backend = "smolvlm2"
            print(f"[JerseyOCR] Backend: SmolVLM2 on {self.device}")
        except Exception as e:
            logger.warning(f"[JerseyOCR] SmolVLM2 initialization skipped ({e}). Using EasyOCR.")
            if EASYOCR_AVAILABLE:
                self._init_easyocr()
            else:
                self.backend = "disabled"

    def _init_easyocr(self):
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
        except:
            gpu_ok = False
        self.vlm = easyocr.Reader(["en"], gpu=gpu_ok, verbose=False)
        self.backend = "easyocr"
        print(f"[JerseyOCR] Backend: EasyOCR (GPU: {gpu_ok})")

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────

    def extract_jersey_numbers(
        self,
        frames: list,
        tracks: list,
        pose_per_frame: list,
        sample_every_n_frames: int = 2,
    ) -> dict:

        if self.backend == "disabled":
            return {}

        votes = defaultdict(lambda: {
            "back": Counter(),
            "front": Counter(),
            "back_w": defaultdict(float),
            "front_w": defaultdict(float),
        })
        confirmed = {}   # track_id → number

        for frame_idx, frame in enumerate(frames):
            if frame_idx % sample_every_n_frames != 0:
                continue

            frame_tracks_list = tracks[frame_idx] if frame_idx < len(tracks) else []
            frame_tracks = {t['track_id']: t for t in frame_tracks_list}
            frame_poses = pose_per_frame[frame_idx] if frame_idx < len(pose_per_frame) else {}

            for track_id, track_data in frame_tracks.items():
                if track_id in confirmed:
                    continue

                bbox = track_data.get("bbox")
                if bbox is None:
                    continue

                pose_data = frame_poses.get(track_id)
                orientation = self._classify_orientation(pose_data)

                if not self._should_attempt_ocr(
                    frame, bbox, pose_data, orientation,
                    track_id, confirmed
                ):
                    continue

                crops_with_meta = self._get_smart_crops(
                    frame, bbox, pose_data, orientation
                )

                for crop, is_back, weight in crops_with_meta:
                    if crop is None or crop.size == 0:
                        continue

                    processed = self._preprocess_crop(crop, is_back)
                    digit, conf = self._run_ocr(processed, orientation)

                    if digit and conf >= self.OCR_CONF_THRESHOLD:
                        weighted_conf = conf * weight
                        side = "back" if is_back else "front"
                        votes[track_id][side][digit] += 1
                        votes[track_id][f"{side}_w"][digit] += weighted_conf
                        print(f"  [JerseyOCR] Frame {frame_idx:04d} | Player #{track_id:02d} | Detected Digit: '{digit}' (Conf: {conf:.2f}, Side: {side})")

                result = self._get_verdict(votes[track_id], early=True)
                if result:
                    confirmed[track_id] = int(result)
                    print(f"  >>> [JerseyOCR] [CONFIRMED] Player #{track_id} locked to Jersey #{result} (Early High Confidence)")

        for track_id, data in votes.items():
            if track_id not in confirmed:
                result = self._get_verdict(data, early=False)
                if result:
                    confirmed[track_id] = int(result)
                    print(f"  >>> [JerseyOCR] [CONFIRMED] Player #{track_id} mapped to Jersey #{result} (Majority Vote)")

        print("\n" + "=" * 65)
        print("  [JerseyOCR] FINAL JERSEY NUMBER RECOGNITION ROSTER")
        print("=" * 65)
        if confirmed:
            for tid, jnum in sorted(confirmed.items()):
                print(f"  * Player Track #{tid:02d}  ===>  Jersey #{jnum:02d} (OCR Verified)")
        else:
            print("  * No high-confidence jersey numbers detected.")
        print("=" * 65 + "\n")

        return {int(k): int(v) for k, v in confirmed.items()}

    # ─────────────────────────────────────────────────────────────────
    # ORIENTATION & GATING
    # ─────────────────────────────────────────────────────────────────

    def _classify_orientation(self, pose_data) -> str:
        if not pose_data:
            return "unknown"

        kps = pose_data.get("keypoints", [])
        if len(kps) <= self.KP_RIGHT_SHOULDER:
            return "unknown"

        ls = kps[self.KP_LEFT_SHOULDER]
        rs = kps[self.KP_RIGHT_SHOULDER]

        ls_conf = ls[2] if len(ls) > 2 else 0
        rs_conf = rs[2] if len(rs) > 2 else 0

        if ls_conf < self.KP_CONF_THRESHOLD or rs_conf < self.KP_CONF_THRESHOLD:
            return "unknown"

        return "back" if ls[0] > rs[0] else "front"

    def _should_attempt_ocr(
        self, frame, bbox, pose_data,
        orientation, track_id, confirmed
    ) -> bool:
        if track_id in confirmed:
            return False

        x1, y1, x2, y2 = map(int, bbox)
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Gate 1: Player size threshold
        if bbox_area < self.MIN_BBOX_AREA:
            return False

        # Gate 2: Blur filter
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if crop.size > 0 and self._is_blurry(crop, threshold=25):
            return False

        return True

    def _is_blurry(self, crop: np.ndarray, threshold: int = 25) -> bool:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    # ─────────────────────────────────────────────────────────────────
    # SMART CROPPING & PREPROCESSING
    # ─────────────────────────────────────────────────────────────────

    def _get_smart_crops(
        self, frame, bbox, pose_data, orientation
    ) -> list:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bh = max(1, y2 - y1)
        bw = max(1, x2 - x1)
        crops = []

        if orientation == "back":
            regions = [
                (0.12, 0.60, 0.15, 0.85),  # standard upper back
                (0.18, 0.52, 0.22, 0.78),  # tight center back
            ]
            for ys, ye, xs, xe in regions:
                cy1 = max(0, int(y1 + bh * ys))
                cy2 = min(h, int(y1 + bh * ye))
                cx1 = max(0, int(x1 + bw * xs))
                cx2 = min(w, int(x1 + bw * xe))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crops.append((crop, True, self.BACK_CROP_WEIGHT))

        elif orientation == "front":
            regions = [
                (0.18, 0.48, 0.05, 0.50),  # chest region
                (0.20, 0.44, 0.08, 0.42),  # tight left chest
            ]
            for ys, ye, xs, xe in regions:
                cy1 = max(0, int(y1 + bh * ys))
                cy2 = min(h, int(y1 + bh * ye))
                cx1 = max(0, int(x1 + bw * xs))
                cx2 = min(w, int(x1 + bw * xe))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crops.append((crop, False, self.FRONT_CROP_WEIGHT))

        else:
            # UNKNOWN: Try torso area
            torso_crop = frame[
                max(0, int(y1 + bh * 0.12)) : min(h, int(y1 + bh * 0.58)),
                max(0, int(x1 + bw * 0.15)) : min(w, int(x1 + bw * 0.85))
            ]
            if torso_crop.size > 0:
                crops.append((torso_crop, True, 1.0))

        return crops

    def _preprocess_crop(
        self, crop: np.ndarray, is_back: bool
    ) -> np.ndarray:
        if crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        # Target minimum height of 120px for clear digit recognition
        target_h = max(120, h * 4)
        scale = target_h / max(h, 1)
        target_w = max(60, int(w * scale))

        crop = cv2.resize(
            crop,
            (target_w, target_h),
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

        # CLAHE adaptive local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Unsharp masking for sharp digit edges
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # ─────────────────────────────────────────────────────────────────
    # OCR BACKENDS
    # ─────────────────────────────────────────────────────────────────

    def _run_ocr(
        self, image: np.ndarray, orientation: str = "unknown"
    ):
        if self.backend == "gemini":
            return self._ocr_gemini(image, orientation)
        elif self.backend == "smolvlm2":
            return self._ocr_smolvlm2(image, orientation)
        elif self.backend == "easyocr":
            return self._ocr_easyocr(image)
        return None, 0.0

    def _ocr_gemini(self, image: np.ndarray, orientation: str):
        self._call_count += 1
        if self._call_count >= 14:
            elapsed = time.time() - self._last_reset
            if elapsed < 60:
                time.sleep(60 - elapsed)
            self._call_count = 0
            self._last_reset = time.time()

        prompts = {
            "back": "Football player BACK. Large jersey number on upper back. Number only (1-99) or unknown.",
            "front": "Football player CHEST. Small jersey number on left chest. Number only (1-99) or unknown.",
            "unknown": "Football player jersey. Find jersey number on back or chest. Number only (1-99) or unknown.",
        }

        pil_img = Image.fromarray(image[:, :, ::-1])
        prompt  = prompts.get(orientation, prompts["unknown"])

        try:
            response = self.vlm.generate_content([pil_img, prompt])
            text     = response.text.strip().translate(self.CHAR_FIXES)
            digits   = re.sub(r"\D", "", text)
            if digits and 1 <= int(digits) <= 99:
                return str(int(digits)), 0.85
        except Exception as e:
            logger.debug(f"[JerseyOCR] Gemini error: {e}")

        return None, 0.0

    def _ocr_smolvlm2(self, image: np.ndarray, orientation: str):
        import torch

        prompts = {
            "back":  "Football player BACK. Large jersey number on upper back. Number only (1-99) or unknown.",
            "front": "Football player CHEST. Small jersey number on left chest. Number only (1-99) or unknown.",
            "unknown": "Football player jersey. Find jersey number on back or chest. Number only (1-99) or unknown.",
        }

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompts.get(orientation, prompts["unknown"])}
            ]
        }]

        pil_img = Image.fromarray(image[:, :, ::-1])
        prompt  = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs  = self.processor(
            text=prompt, images=[pil_img], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.vlm.generate(**inputs, max_new_tokens=5)

        text   = self.processor.decode(
            output[0], skip_special_tokens=True
        ).strip().translate(self.CHAR_FIXES)
        digits = re.sub(r"\D", "", text)

        if digits and 1 <= int(digits) <= 99:
            return str(int(digits)), 0.88

        return None, 0.0

    def _ocr_easyocr(self, image: np.ndarray):
        try:
            # Allowlist digits for high speed and accuracy
            results = self.vlm.readtext(image, detail=1, allowlist="0123456789")
            best_digit = None
            best_conf = 0.0
            for (_, text, conf) in results:
                digits = re.sub(r"\D", "", text)
                if digits and 1 <= int(digits) <= 99 and conf > best_conf:
                    best_digit = str(int(digits))
                    best_conf = float(conf)
            if best_digit and best_conf >= self.OCR_CONF_THRESHOLD:
                return best_digit, best_conf
        except Exception as e:
            logger.debug(f"[JerseyOCR] EasyOCR error: {e}")
        return None, 0.0

    # ─────────────────────────────────────────────────────────────────
    # VOTING & VERDICT
    # ─────────────────────────────────────────────────────────────────

    def _get_verdict(self, data: dict, early: bool = False) -> str:
        combined = Counter()
        combined.update(data["back"])
        combined.update(data["front"])

        if not combined:
            return None

        winner, winner_votes = combined.most_common(1)[0]
        total_votes = sum(combined.values())
        share = winner_votes / max(total_votes, 1)

        min_req = 2 if early else self.MIN_VOTES
        if winner_votes >= min_req and share >= self.MIN_VOTE_SHARE:
            return winner

        return None
