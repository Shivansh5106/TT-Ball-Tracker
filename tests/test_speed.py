"""Speed calculation on synthetic positions/timestamps, including outlier
and gap rejection, and the measured-only evidence rule."""
import math

import numpy as np
import pytest

from src.analysis import SpeedParams, compute_speeds
from src.calibration import TableCalibration
from src.tracking import TrackPoint

# Image square where 100 px = 10 m -> 10 px per metre.
CAL = TableCalibration([[0, 0], [100, 0], [100, 100], [0, 100]],
                       length_m=10, width_m=10)
FPS = 30.0


def rec(frame, x, y, status="detected"):
    return TrackPoint(frame=frame, t=frame / FPS, status=status,
                      meas_x=x, meas_y=y, x=x, y=y)


def test_constant_motion_gives_expected_speed():
    records = [rec(i, 5 + 5 * i, 50) for i in range(10)]  # 5 px = 0.5 m /frame
    speeds, speeds_px, _ = compute_speeds(records, CAL)
    fin = speeds[np.isfinite(speeds)]
    assert fin.size == 9
    assert np.median(fin) == pytest.approx(0.5 * FPS, rel=1e-6)  # 15 m/s
    px = speeds_px[np.isfinite(speeds_px)]
    assert np.median(px) == pytest.approx(5 * FPS, rel=1e-6)    # 150 px/s


def test_uncalibrated_reports_pixel_speeds_only():
    records = [rec(i, 5 + 5 * i, 50) for i in range(10)]
    speeds, speeds_px, _ = compute_speeds(records, None)
    assert np.all(np.isnan(speeds))
    assert np.any(np.isfinite(speeds_px))


def test_outlier_spike_is_rejected():
    records = [rec(i, 5 + 5 * i, 50) for i in range(12)]
    records[6] = rec(6, 500, 50)  # single wild measurement
    speeds, _, outliers = compute_speeds(records, CAL, SpeedParams())
    assert np.any(outliers)
    assert math.isnan(speeds[6]) or math.isnan(speeds[7])
    fin = speeds[np.isfinite(speeds)]
    assert np.median(fin) == pytest.approx(15.0, rel=0.02)


def test_no_speed_across_long_gap():
    records = [rec(i, 5 + 5 * i, 50) for i in range(15)]
    for i in range(5, 11):  # tracking lost for a while -> not measured
        records[i] = rec(i, math.nan, math.nan, status="lost")
    speeds, _, _ = compute_speeds(records, CAL, SpeedParams(max_gap_frames=4))
    assert math.isnan(speeds[11])  # 5-frame gap exceeds the limit
    assert speeds[4] == pytest.approx(15.0, rel=1e-6)
    assert math.isnan(speeds[5])


def test_predicted_points_are_not_speed_evidence():
    # Measured at frames 0 and 2; frame 1 is Kalman-coasted with a wild
    # position that must not create a speed estimate.
    records = [rec(0, 10, 50), rec(1, 999, 999, status="predicted"),
               rec(2, 20, 50)]
    speeds, speeds_px, _ = compute_speeds(records, CAL)
    assert math.isnan(speeds[1])
    assert speeds[2] == pytest.approx(1.0 / (2.0 / FPS), rel=0.05)  # 15 m/s
    assert math.isnan(speeds_px[1])
