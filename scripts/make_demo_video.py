"""Generate a clearly-synthetic demo clip so the app can be exercised
without a real recording.

Usage:
    python scripts/make_demo_video.py [output.mp4]

The clip shows a dark room, a table surface and an orange ball bouncing
twice.  It is generated data for plumbing/UI checks only — it says nothing
about real-world accuracy.  Suggested calibration corners for this clip
(the table top corners, in the documented clockwise order):
    Top-left (130, 105)  Top-right (510, 105)
    Bottom-right (620, 175)  Bottom-left (20, 175)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 360, 30, 8
BOUNCES = (2.0, 5.0)          # seconds
TABLE_TOP = 175.0             # image y of the playing surface
FLOOR = 300.0


def ball_y(t: float) -> float:
    # Parabolic arcs between bounces, ~g/8 scaled for a visible arc.
    y = TABLE_TOP
    last = 0.0
    for tb in BOUNCES + (float(SECONDS) + 1,):
        if last <= t <= tb:
            span = max(tb - last, 1e-3)
            u = (t - last) / span
            y = TABLE_TOP - 4.0 * span * 30 * u * (1 - u)  # apex mid-arc
            break
        last = tb
    return y


def main(out: str) -> None:
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not open the video writer.")
    rng = np.random.default_rng(3)
    for f in range(int(FPS * SECONDS)):
        t = f / FPS
        frame = np.full((H, W, 3), 26, np.uint8)
        cv2.rectangle(frame, (20, int(TABLE_TOP)), (620, int(FLOOR)),
                      (48, 52, 58), -1)                    # table body
        cv2.rectangle(frame, (130, 100), (510, 106), (70, 75, 80), -1)
        cv2.line(frame, (285, 96), (355, 96), (120, 120, 125), 3)  # net hint
        x = 60 + (W - 120) * (0.5 - 0.5 * np.cos(2 * np.pi * t / 7.0))
        y = ball_y(t)
        c = (int(round(x + rng.normal(0, 0.4))), int(round(y - 7 + rng.normal(0, 0.4))))
        cv2.circle(frame, c, 7, (0, 110, 255), -1)         # orange ball
        writer.write(frame)
    writer.release()
    print(f"wrote {out} ({FPS} fps, {SECONDS} s, {W}x{H})")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[1] / "demo_clip.mp4")
    main(dest)
