"""Streamlit UI regression tests via the AppTest harness.

These execute the app script body with a seeded "uploaded" video and click
the Preview button — catching errors that only appear once results render
(e.g. a helper used before its definition), which empty-state smoke tests
cannot see.
"""
import numpy as np

from conftest import FPS
from src.analysis import VideoInfo, probe_video

from streamlit.testing.v1 import AppTest

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")


def _seed(at: AppTest, video_path: str) -> None:
    info = VideoInfo(
        path=video_path, width=320, height=240, fps=FPS, frame_count=90,
        duration_s=3.0, timestamp_source="frame-index/fps",
        fps_is_default=False, frame_count_is_estimate=False, warnings=[])
    at.session_state["video_path"] = video_path
    at.session_state["video_info"] = info
    at.session_state["upload_name"] = "synthetic.mp4"
    at.session_state["upload_size"] = 12345
    at.session_state["rep_frame"] = np.zeros((240, 320, 3), np.uint8)


def test_empty_state_renders_cleanly():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error


def test_preview_run_has_no_errors(synthetic_video):
    at = AppTest.from_file(APP, default_timeout=180)
    _seed(at, synthetic_video)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    buttons = [b for b in at.button if b.label and "Run preview" in b.label]
    assert buttons, "preview button not rendered"
    buttons[0].click()
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    assert "preview" in at.session_state, "preview result not stored"
    preview = at.session_state["preview"]
    assert preview.processed_frames > 0
    assert preview.measured_coverage() > 0.5


def test_white_preset_callback_updates_hsv():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].set_value("white").run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["det_val_low"] == 175
    assert at.session_state["det_sat_high"] == 100
    assert at.session_state["det_min_motion"] == 0.15


def test_auto_mode_resolves_colour_and_previews(synthetic_video):
    """Auto mode must pick the orange preset on the synthetic orange-ball
    clip and produce a working preview without errors."""
    at = AppTest.from_file(APP, default_timeout=180)
    _seed(at, synthetic_video)
    at.run()
    at.sidebar.selectbox[0].set_value("auto").run()
    assert not at.exception, [str(e.value) for e in at.exception]

    buttons = [b for b in at.button if b.label and "Run preview" in b.label]
    assert buttons
    buttons[0].click()
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    try:
        resolved = at.session_state["resolved_color"]
    except (KeyError, AttributeError):
        resolved = None
    assert resolved == "orange"
    preview = at.session_state["preview"]
    assert preview.processed_frames > 0
    assert preview.measured_coverage() > 0.5
