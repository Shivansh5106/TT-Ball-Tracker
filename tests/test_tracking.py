"""Tracker behaviour: follow, coast, reset after missed-frame limit,
controlled reacquisition — plus Kalman frame-rate awareness."""
import math

import pytest

from src.detection import Candidate
from src.tracking import BallTracker, KalmanCV, TrackingParams

FPS = 30.0


def cand(x, y, score=0.8):
    return Candidate(x=x, y=y, radius=5.0, area=80.0, circularity=0.9,
                     solidity=0.95, motion_overlap=1.0, score=score)


def test_follows_straight_line_all_detected():
    tr = BallTracker(TrackingParams(reacq_confirm=1), FPS)
    statuses = []
    for i in range(30):
        rec = tr.step([cand(50 + 4 * i, 60)], 1.0 / FPS, i, i / FPS)
        statuses.append(rec.status)
    assert all(s == "detected" for s in statuses)
    last = tr.step([cand(50 + 4 * 30, 60)], 1.0 / FPS, 30, 1.0)
    assert last.status == "detected"
    assert last.x == pytest.approx(170, abs=3)


def test_coast_then_reset_after_missed_frame_limit():
    tr = BallTracker(TrackingParams(max_missed_frames=3, reacq_confirm=1), FPS)
    for i in range(20):
        tr.step([cand(50 + 4 * i, 60)], 1.0 / FPS, i, i / FPS)
    assert tr.is_tracking
    statuses = [tr.step([], 1.0 / FPS, 20 + k, (20 + k) / FPS).status
                for k in range(6)]
    assert statuses[:3] == ["predicted"] * 3     # coast while within limit
    assert all(s == "lost" for s in statuses[3:])  # reset afterwards
    assert not tr.is_tracking


def test_reacquisition_requires_confirmation():
    tr = BallTracker(TrackingParams(max_missed_frames=1, reacq_confirm=3), FPS)
    for i in range(10):
        tr.step([cand(50 + 4 * i, 60)], 1.0 / FPS, i, i / FPS)
    tr.step([], 1.0 / FPS, 10, 10 / FPS)
    tr.step([], 1.0 / FPS, 11, 11 / FPS)
    assert not tr.is_tracking
    # Single stray frame must not re-lock the filter.
    tr.step([cand(400, 400)], 1.0 / FPS, 12, 12 / FPS)
    assert not tr.is_tracking
    for k in range(4):
        rec = tr.step([cand(120, 60)], 1.0 / FPS, 13 + k, (13 + k) / FPS)
        assert rec.status == "detected"  # raw measurement is honest evidence
    assert tr.is_tracking


def test_gating_ignores_candidates_far_from_prediction():
    tr = BallTracker(TrackingParams(max_missed_frames=10, max_jump_px=40,
                                    reacq_confirm=1), FPS)
    for i in range(15):
        tr.step([cand(100 + 3 * i, 80)], 1.0 / FPS, i, i / FPS)
    rec = tr.step([cand(600, 500)], 1.0 / FPS, 15, 0.5)  # far away -> ignored
    assert rec.status == "predicted"
    assert rec.x < 200


def test_kalman_prediction_scales_with_dt():
    kf = KalmanCV(100, 100, process_noise=2500.0)
    kf.predict(1.0 / FPS)
    kf.correct(106, 100)  # ~6 px per 1/30 s -> 180 px/s
    kf.predict(1.0 / FPS)
    kf.correct(112, 100)
    kf.predict(1.0 / FPS)
    kf.correct(118, 100)
    vx, vy = kf.vel
    assert vx == pytest.approx(180, rel=0.25)
    assert vy == pytest.approx(0, abs=40)
    state = kf.predict(0.5 / FPS)  # half-interval step
    assert float(state[0, 0]) == pytest.approx(118 + vx * 0.5 / FPS, rel=0.05)
