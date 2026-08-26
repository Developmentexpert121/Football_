import os
import sys
import cv2
import argparse
import numpy as np

from main import run_pipeline

def generate_synthetic_demo_video(output_path: str = "data/input_videos/match_01.mp4", duration_sec: int = 6, fps: int = 25):
    """
    Generates a synthetic 1280x720 football match video clip (pitch background + moving players + ball)
    for immediate pipeline testing without external match downloads.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    num_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Generating synthetic football match demo video ({num_frames} frames)...")

    # Define synthetic moving entities
    # Player 1 (Team A - Red): moving right
    # Player 2 (Team B - Blue): moving left towards Player 1
    # Player 3 (Ref - Yellow): static
    # Ball (Cyan): rolling between players

    for f in range(num_frames):
        t = f / num_frames

        # Create Green Pitch Background with lines
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8) # Pitch green
        cv2.rectangle(frame, (100, 50), (1180, 670), (255, 255, 255), 3) # Touchline
        cv2.line(frame, (640, 50), (640, 670), (255, 255, 255), 2)       # Halfway line
        cv2.circle(frame, (640, 360), 80, (255, 255, 255), 2)            # Center circle
        cv2.rectangle(frame, (100, 200), (300, 520), (255, 255, 255), 2) # Left penalty area
        cv2.rectangle(frame, (980, 200), (1180, 520), (255, 255, 255), 2)# Right penalty area

        # Entity 1: Player 1 (Team A Red)
        p1_x = int(350 + t * 300)
        p1_y = int(300 + np.sin(t * np.pi * 2) * 40)
        cv2.rectangle(frame, (p1_x - 15, p1_y - 40), (p1_x + 15, p1_y + 40), (50, 50, 220), -1) # Red jersey

        # Entity 2: Player 2 (Team B Blue)
        p2_x = int(700 - t * 250)
        p2_y = int(320 - np.sin(t * np.pi * 2) * 30)
        cv2.rectangle(frame, (p2_x - 15, p2_y - 40), (p2_x + 15, p2_y + 40), (220, 100, 50), -1) # Blue jersey

        # Entity 3: Referee (Yellow)
        cv2.rectangle(frame, (600, 250, 30, 80), (0, 220, 220), -1)

        # Entity 4: Ball
        ball_x = int(p1_x + 30 + t * 40)
        ball_y = int(p1_y + 20)
        cv2.circle(frame, (ball_x, ball_y), 10, (255, 255, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), 10, (0, 0, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Synthetic demo video created successfully at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Match Video Analysis System Runner")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic match video clip and run full demo pipeline")
    parser.add_argument("--input", type=str, default="data/input_videos/match_01.mp4", help="Path to input match video")
    parser.add_argument("--output", type=str, default="data/output_videos/match_01_annotated.mp4", help="Path to output annotated video")
    parser.add_argument("--no-stubs", action="store_true", help="Bypass cached stubs")

    args = parser.parse_args()

    input_path = args.input
    if args.demo or not os.path.exists(input_path):
        if not os.path.exists(input_path):
            print(f"Input video '{input_path}' not found. Automatically generating synthetic demo video...")
        generate_synthetic_demo_video(input_path)

    run_pipeline(
        input_video_path=input_path,
        output_video_path=args.output,
        use_stubs=not args.no_stubs
    )
