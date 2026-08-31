# recalibrate_homography.py
# Native Tkinter GUI Window Calibrator — Opens the real video frame in a window

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

VIDEO_PATH = "data/input_videos/match_01.mp4"
FRAME_NUMBER = 100   # Frame number where full pitch is visible

# Real-world pitch corners in metres: TL, TR, BR, BL
PITCH_PTS = np.float32([
    [0.0,   0.0],   # Top-Left corner
    [105.0, 0.0],   # Top-Right corner
    [105.0, 68.0],  # Bottom-Right corner
    [0.0,   68.0],  # Bottom-Left corner
])

class HomographyCalibratorGUI:
    def __init__(self, frame_bgr):
        self.frame_bgr = frame_bgr
        self.h, self.w = frame_bgr.shape[:2]
        self.clicked_pts = []
        self.labels = ['TL (Top-Left)', 'TR (Top-Right)', 'BR (Bottom-Right)', 'BL (Bottom-Left)']

        # Create main Tkinter GUI window
        self.root = tk.Tk()
        self.root.title("Football Homography Calibrator — Real Video Frame")
        self.root.geometry(f"{min(self.w, 1280)}x{min(self.h + 80, 800)}")

        # Convert OpenCV BGR -> PIL Image -> ImageTk PhotoImage
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(frame_rgb)
        
        # Scale factor if frame is larger than screen width 1280
        self.scale = 1.0
        max_disp_w, max_disp_h = 1280, 720
        if self.w > max_disp_w or self.h > max_disp_h:
            self.scale = min(max_disp_w / self.w, max_disp_h / self.h)
            disp_w, disp_h = int(self.w * self.scale), int(self.h * self.scale)
            disp_img = self.pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        else:
            disp_img = self.pil_img

        self.tk_img = ImageTk.PhotoImage(disp_img)

        # Header instruction label
        self.instruction_var = tk.StringVar()
        self.instruction_var.set("Click Corner 1 of 4: Top-Left (TL)")
        self.lbl_instruction = tk.Label(
            self.root, textvariable=self.instruction_var,
            font=("Segoe UI", 12, "bold"), bg="#1e1e1e", fg="#00ff66", pady=8
        )
        self.lbl_instruction.pack(fill=tk.X)

        # Canvas displaying the real video frame
        self.canvas = tk.Canvas(self.root, width=disp_img.width, height=disp_img.height, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)

        # Bind mouse click event
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def on_canvas_click(self, event):
        if len(self.clicked_pts) >= 4:
            return

        # Map display canvas coordinates back to original video frame coordinates
        real_x = round(event.x / self.scale, 1)
        real_y = round(event.y / self.scale, 1)
        self.clicked_pts.append([real_x, real_y])

        # Draw point on canvas
        r = 6
        color = "#00ff00"
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill=color, outline="white", width=2)
        idx = len(self.clicked_pts) - 1
        tag = ['TL', 'TR', 'BR', 'BL'][idx]
        self.canvas.create_text(event.x + 12, event.y - 10, text=tag, fill="#00ff00", font=("Segoe UI", 11, "bold"))

        print(f"  [+] Clicked Point {idx + 1} ({tag}): Pixel [{real_x}, {real_y}]")

        if len(self.clicked_pts) < 4:
            next_label = self.labels[len(self.clicked_pts)]
            self.instruction_var.set(f"Click Corner {len(self.clicked_pts) + 1} of 4: {next_label}")
        else:
            self.instruction_var.set("✓ All 4 Corners Selected! Calculating Homography...")
            self.verify_and_output()

    def verify_and_output(self):
        img_pts = np.float32(self.clicked_pts)
        H, _ = cv2.findHomography(img_pts, PITCH_PTS)

        # Test left goal post bases
        test_pixels = np.float32([[[180, 580]], [[480, 420]]])
        mapped = cv2.perspectiveTransform(test_pixels, H)

        print("\n" + "=" * 65)
        print("  HOMOGRAPHY CALIBRATION SUCCESSFUL")
        print("=" * 65)
        print(f"  Left Goal Post P1 [180, 580] -> X={mapped[0][0][0]:.2f}m, Y={mapped[0][0][1]:.2f}m")
        print(f"  Left Goal Post P2 [480, 420] -> X={mapped[1][0][0]:.2f}m, Y={mapped[1][0][1]:.2f}m")
        print("  Expected:                       X ~= 0m, Y between 30.34m - 37.66m")
        print("-" * 65)
        print("\nPaste these values into config.yaml under 'pitch.reference_points_image':\n")
        print("    reference_points_image:")
        for p in self.clicked_pts:
            print(f"      - [{int(round(p[0]))}, {int(round(p[1]))}]")
        print("\n" + "=" * 65 + "\n")

        messagebox.showinfo("Calibration Complete", "4 Pitch Corners Saved!\nCheck terminal output for config.yaml values.")
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"ERROR: Could not read frame {FRAME_NUMBER} from '{VIDEO_PATH}'.")
        return

    print("Opening real video frame in Tkinter GUI window...")
    gui = HomographyCalibratorGUI(frame)
    gui.run()

if __name__ == "__main__":
    main()