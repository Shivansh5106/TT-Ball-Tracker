import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H, FPS, N = 320, 240, 30.0, 90
BOUNCE_FRAME = 45


def make_video(path) -> None:
    """Synthetic clip: dark room, table strip, one orange ball with a bounce."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    for f in range(N):
        frame = np.full((H, W, 3), 28, np.uint8)
        cv2.rectangle(frame, (20, 182), (300, 232), (55, 55, 55), -1)
        y = 182 - 2.6 * abs(f - BOUNCE_FRAME)
        x = 40 + 2.7 * f
        centre = (int(round(x + rng.normal(0, 0.3))),
                  int(round(y + rng.normal(0, 0.3))))
        cv2.circle(frame, centre, 6, (0, 120, 255), -1)  # orange ball, BGR
        cv2.circle(frame, centre, 6, (90, 170, 255), 1)
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("synth") / "synthetic.mp4"
    make_video(p)
    return str(p)
