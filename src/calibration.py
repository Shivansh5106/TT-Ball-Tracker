"""Table calibration via homography from image pixels to table coordinates.

The four table corners are clicked in a documented clockwise order and mapped
to a rectangle of `length_m x width_m` metres.  The resulting homography is
exact for points ON the table plane.  A ball in flight lies above that plane,
so its projected position is a shadow-like approximation — speeds derived from
projections are reported as "approximate table-plane projected speed", never
as true 3D ball speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

DEFAULT_TABLE_LENGTH_M = 2.74
DEFAULT_TABLE_WIDTH_M = 1.525

# Clockwise order as seen in the image.  The top edge (corner 1 -> 2) should
# be the edge the camera associates with the table's LONG side; x runs along
# the table length, y along its width.
CORNER_ORDER = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
CORNER_GUIDE = {
    "Top-left": "Upper-most left corner of the table surface in the image",
    "Top-right": "Upper-most right corner of the table surface in the image",
    "Bottom-right": "Lower-most right corner of the table surface in the image",
    "Bottom-left": "Lower-most left corner of the table surface in the image",
}


class CalibrationError(ValueError):
    pass


def validate_corners(image_points: Sequence[Sequence[float]],
                     min_separation_px: float = 5.0) -> np.ndarray:
    pts = np.asarray(image_points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise CalibrationError("Exactly four (x, y) corner points are required.")
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.hypot(*(pts[i] - pts[j]))) < min_separation_px:
                raise CalibrationError(
                    "Corners are too close together or duplicated; give four "
                    "distinct points spread across the table.")
    for k in range(4):
        a, b, c = pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1.0:  # any three corners nearly collinear
            raise CalibrationError(
                "Three or more corners are (nearly) collinear; the homography "
                "would be degenerate. Pick the four corners of a proper quad.")
    return pts


@dataclass
class TableCalibration:
    image_points: List[List[float]]
    length_m: float = DEFAULT_TABLE_LENGTH_M
    width_m: float = DEFAULT_TABLE_WIDTH_M
    _homography: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        pts = validate_corners(self.image_points)
        if self.length_m <= 0 or self.width_m <= 0:
            raise CalibrationError("Table length and width must be positive.")
        dst = np.array([[0, 0], [self.length_m, 0],
                        [self.length_m, self.width_m], [0, self.width_m]],
                       dtype=np.float64)
        h, mask = cv2.findHomography(pts, dst)
        if h is None or int(np.sum(mask)) < 4:
            raise CalibrationError(
                "Could not compute a valid homography from these corners.")
        self._homography = h
        self.image_points = pts.tolist()

    # ------------------------------------------------------------- geometry
    def project(self, x: float, y: float) -> Tuple[float, float]:
        p = np.array([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(p, self._homography)[0, 0]
        return float(out[0]), float(out[1])

    def project_many(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(xy) == 0:
            return np.array([]), np.array([])
        out = cv2.perspectiveTransform(
            np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2),
            self._homography).reshape(-1, 2)
        return out[:, 0].copy(), out[:, 1].copy()

    def inside_table(self, x_m: float, y_m: float,
                     margin_m: float = 0.0) -> bool:
        return (-margin_m <= x_m <= self.length_m + margin_m
                and -margin_m <= y_m <= self.width_m + margin_m)

    def table_quad_image(self, n: int = 60) -> np.ndarray:
        """Dense polygon of the table rectangle back in image pixels (UI)."""
        xs = np.linspace(0, self.length_m, n)
        ys = np.linspace(0, self.width_m, n)
        boundary = ([(x, 0.0) for x in xs] + [(self.length_m, y) for y in ys]
                    + [(x, self.width_m) for x in xs[::-1]]
                    + [(0.0, y) for y in ys[::-1]])
        inv = np.linalg.inv(self._homography)
        pts = cv2.perspectiveTransform(
            np.array(boundary, dtype=np.float64).reshape(-1, 1, 2),
            inv).reshape(-1, 2)
        return np.round(pts).astype(np.int32)

    def to_dict(self) -> dict:
        return {
            "corner_order": list(CORNER_ORDER),
            "image_points_px": self.image_points,
            "table_length_m": self.length_m,
            "table_width_m": self.width_m,
            "note": ("Homography is exact on the table plane only; airborne "
                     "positions project approximately (shadow point)."),
        }
