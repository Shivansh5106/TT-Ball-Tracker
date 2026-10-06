"""End-to-end synthetic check: generate a small video with a bouncing orange
ball on a static background, run the full streaming pipeline and the video
annotation pass.  This verifies wiring, not real-world accuracy."""
import math

import cv2
import numpy as np
import pytest

from conftest import BOUNCE_FRAME, FPS, H, N, W
from src.analysis import VideoAnalyzer, compute_speeds, probe_video
from src.bounce import BounceParams
from src.detection import DetectionParams
from src.export import annotate_video
from src.tracking import TrackingParams


def test_probe_reports_metadata(synthetic_video):
    info = probe_video(synthetic_video)
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert info.frame_count == pytest.approx(N, abs=3)
    assert info.timestamp_source in ("container-msec", "frame-index/fps")


def test_full_pipeline_detects_tracks_and_bounces(synthetic_video):
    det = DetectionParams(ball_color="orange")
    det.apply_preset()
    result = VideoAnalyzer(
        synthetic_video, detection=det,
        tracking=TrackingParams(max_missed_frames=6, max_jump_px=120),
        bounce=BounceParams(cooldown_frames=10),
    ).analyze()

    assert result.processed_frames == N
    coverage = result.measured_coverage()
    assert coverage > 0.8, f"coverage too low: {coverage:.2f}"
    assert result.bounces, "expected at least one estimated bounce"
    assert abs(result.bounces[0].frame - BOUNCE_FRAME) <= 6

    speeds, speeds_px, _ = compute_speeds(result.records, None,
                                          result.speed_params)
    fin = speeds_px[np.isfinite(speeds_px)]
    expected = math.hypot(2.7, 2.6) * FPS  # ~111 px/s
    assert fin.size > N * 0.5
    assert np.median(fin) == pytest.approx(expected, rel=0.35)
    assert np.all(np.isnan(speeds))  # uncalibrated -> no metric speeds


def test_annotated_video_written(synthetic_video, tmp_path):
    det = DetectionParams(ball_color="orange")
    det.apply_preset()
    result = VideoAnalyzer(synthetic_video, detection=det).analyze()
    out = tmp_path / "annotated.mp4"
    annotate_video(synthetic_video, str(out), result)
    assert out.exists() and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None and frame.shape[:2] == (H, W)
    finally:
        cap.release()
