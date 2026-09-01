"""
Adapter script for abhishek-dev branch to run GoalDetector on match videos
and output detection statistics and timestamp logs.
"""
import os
import sys
import argparse
from src.abhishek_goal_detector import GoalDetector

def run_abhishek_detector(video_path: str, model_path: str = None, conf: float = 0.25, debug: bool = True):
    if not os.path.exists(video_path):
        print(f"[Error] Video path not found: {video_path}")
        return []

    print(f"\n==================================================")
    print(f"RUNNING ABHISHEK-DEV GOAL DETECTOR ON: {video_path}")
    print(f"==================================================")

    detector = GoalDetector(
        model_path=model_path,
        conf_threshold=conf,
        video_path=video_path,
        debug_mode=debug,
        visualization=False
    )

    detected_goals = detector.process_video()
    detector.display_results()
    return detected_goals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GoalDetector on match video')
    parser.add_argument('--video', type=str, default='data/input_videos/match_01.mp4', help='Video file path')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    run_abhishek_detector(args.video, args.model, args.conf)
