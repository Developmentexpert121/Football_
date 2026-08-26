import cv2
import os
from typing import List, Tuple, Generator, Optional
import numpy as np

class VideoIO:
    """
    Handles video frame reading, resizing, and video writing operations.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found at: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frames(self, target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Reads all frames from the video into memory.
        Optionally resizes frames to target_size (width, height).
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if target_size:
                frame = cv2.resize(frame, target_size)
            frames.append(frame)
        return frames

    def stream_frames(self, target_size: Optional[Tuple[int, int]] = None) -> Generator[np.ndarray, None, None]:
        """
        Yields frames one by one for streaming execution.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            if target_size:
                frame = cv2.resize(frame, target_size)
            yield frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    @staticmethod
    def save_video(output_path: str, frames: List[np.ndarray], fps: float = 25.0):
        """
        Saves a list of frames to an MP4 video file.
        """
        if not frames:
            print("Warning: No frames to save.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            writer.write(frame)

        writer.release()
        print(f"Video saved successfully to: {output_path}")
