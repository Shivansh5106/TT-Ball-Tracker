"""Synthetic homography tests (no real video involved)."""
import numpy as np
import pytest

from src.calibration import (CalibrationError, TableCalibration,
                             validate_corners)


def test_unit_square_projection_is_identity():
    cal = TableCalibration([[0, 0], [100, 0], [100, 100], [0, 100]],
                           length_m=100, width_m=100)
    for px, py in [(0, 0), (100, 100), (50, 50), (25, 75)]:
        x, y = cal.project(px, py)
        assert x == pytest.approx(px, abs=1e-6)
        assert y == pytest.approx(py, abs=1e-6)


def test_table_mapping_known_corners():
    # Image square maps to a 2.74 x 1.525 m table; corners and centre known.
    cal = TableCalibration([[100, 100], [300, 100], [300, 300], [100, 300]],
                           length_m=2.74, width_m=1.525)
    assert cal.project(100, 100) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert cal.project(300, 100) == pytest.approx((2.74, 0.0), abs=1e-6)
    assert cal.project(300, 300) == pytest.approx((2.74, 1.525), abs=1e-6)
    assert cal.project(200, 200) == pytest.approx((1.37, 0.7625), abs=1e-3)
    assert cal.inside_table(1.0, 0.5)
    assert not cal.inside_table(5.0, 0.5)
    assert cal.inside_table(2.9, 0.5, margin_m=0.35)


def test_project_many_matches_single():
    cal = TableCalibration([[0, 0], [200, 0], [200, 100], [0, 100]],
                           length_m=2, width_m=1)
    pts = np.array([[0, 0], [200, 100], [100, 50]])
    xs, ys = cal.project_many(pts)
    for (px, py), x, y in zip(pts, xs, ys):
        assert (x, y) == pytest.approx(cal.project(px, py), abs=1e-6)


def test_collinear_corners_rejected():
    with pytest.raises(CalibrationError):
        TableCalibration([[0, 0], [100, 100], [200, 200], [0, 100]])


def test_duplicate_or_tiny_corners_rejected():
    with pytest.raises(CalibrationError):
        TableCalibration([[10, 10], [10, 10], [200, 200], [10, 300]])
    with pytest.raises(CalibrationError):
        validate_corners([[0, 0], [1, 1], [200, 0], [0, 200]])


def test_wrong_point_count_rejected():
    with pytest.raises(CalibrationError):
        TableCalibration([[0, 0], [100, 0], [0, 100]], 2, 1)
