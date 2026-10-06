"""Bounce heuristic on synthetic trajectories: reversal detection, cooldown
duplicate suppression, gap rejection and measured-only evidence."""
import math

import pytest

from src.bounce import BounceParams, detect_bounces
from src.calibration import TableCalibration
from src.tracking import TrackPoint

FPS = 30.0
CAL = TableCalibration([[100, 100], [400, 100], [400, 350], [100, 350]],
                       length_m=2.74, width_m=1.525)


def traj(bounce_frames, n=70, vy=3.0, vx=2.0, y_low=200.0):
    """Ball in image coords (y down). Each bounce frame is a y-maximum:
    the ball falls (vy>0), contacts the table at tb, then rises (vy<0)."""
    records = []
    for f in range(n):
        if not bounce_frames:
            y = 150.0 + 1.5 * f
        else:
            y = max(y_low - vy * abs(f - tb) for tb in bounce_frames)
        x = 150 + vx * f
        y = min(y, 340.0)
        records.append(TrackPoint(frame=f, t=f / FPS, status="detected",
                                  meas_x=x, meas_y=y, x=x, y=y))
    return records


def test_single_bounce_detected_near_event():
    events = detect_bounces(traj([30]), None)
    assert len(events) == 1
    assert abs(events[0].frame - 30) <= 2
    assert events[0].confidence > 0.3
    assert math.isnan(events[0].table_x)  # uncalibrated -> no table coords


def test_cooldown_suppresses_close_duplicates():
    # Two reversals 5 frames apart (< cooldown 10): one physical-looking
    # cluster must produce exactly one event.
    events = detect_bounces(traj([30, 35]), None)
    assert len(events) == 1


def test_well_separated_bounces_both_counted():
    events = detect_bounces(traj([20, 50]), None)
    assert len(events) == 2
    assert events[0].frame < events[1].frame


def test_cooldown_threshold_controls_merge_distance():
    # Bounces 12 frames apart resolve as two events (~14 frames between
    # estimated events); a cooldown longer than that merges them into one.
    # (Bounces only ~5 frames apart cannot be resolved at all — see README
    # limitations.)
    assert len(detect_bounces(traj([20, 32]), None,
                              BounceParams(cooldown_frames=12))) == 2
    assert len(detect_bounces(traj([20, 32]), None,
                              BounceParams(cooldown_frames=18))) == 1


def test_tracking_gap_rejects_event():
    records = traj([30])
    for f in range(26, 35):  # occlusion right at the bounce
        records[f] = TrackPoint(frame=f, t=f / FPS, status="lost")
    assert detect_bounces(records, None) == []


def test_no_measured_points_no_events():
    records = [TrackPoint(frame=f, t=f / FPS, status="predicted",
                          x=150 + 2 * f, y=200)
               for f in range(60)]
    assert detect_bounces(records, None) == []


def test_calibrated_far_from_table_rejected():
    # Same reversal shape but the ball sits well outside the table quad.
    records = traj([30])
    for r in records:
        r.meas_x, r.meas_y = r.meas_x + 500, r.meas_y
    assert detect_bounces(records, CAL) == []


def test_calibrated_on_table_is_confident_and_projected():
    events = detect_bounces(traj([30]), CAL)
    assert len(events) == 1
    assert math.isfinite(events[0].table_x)
    assert CAL.inside_table(events[0].table_x, events[0].table_y, 0.35)
    uncal = detect_bounces(traj([30]), None)[0]
    assert events[0].confidence > uncal.confidence


def test_weak_reversal_ignored():
    # Correct bounce shape but far below the reversal-strength threshold.
    records = []
    for f in range(60):
        y = 200 - 0.3 * abs(f - 30)
        records.append(TrackPoint(frame=f, t=f / FPS, status="detected",
                                  meas_x=150 + 2 * f, meas_y=y,
                                  x=150 + 2 * f, y=y))
    assert detect_bounces(records, None) == []
