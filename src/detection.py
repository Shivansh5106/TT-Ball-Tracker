"""Ball candidate detection using classical image processing.

Per-frame pipeline:
  1. HSV colour segmentation (white or orange ball, adjustable thresholds).
  2. Motion evidence from frame differencing or MOG2 background subtraction.
  3. Morphological cleanup (open + close with elliptical kernels).
  4. Contour filtering (area, circularity, equivalent radius).
  5. Candidate scoring that combines appearance (circularity, solidity),
     motion overlap and distance to the Kalman-predicted location.

The largest contour is never blindly selected: every surviving candidate is
scored and ranked, so multiple simultaneous candidates are handled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Sensible starting thresholds; the user can tune everything in the UI.
# The white preset assumes motion-blurred footage (value cut at 175, not 190)
# and hard-gates on motion overlap so static look-alikes (table lines, banner
# text, shoes at rest) are rejected outright.
WHITE_PRESET = dict(hue_low=0, hue_high=179, sat_low=0, sat_high=100,
                    val_low=175, val_high=255,
                    min_circularity=0.30, min_motion_overlap=0.15,
                    max_area_px=2500, max_radius_px=30)
ORANGE_PRESET = dict(hue_low=5, hue_high=25, sat_low=90, sat_high=255,
                     val_low=120, val_high=255,
                     min_circularity=0.35, min_motion_overlap=0.15,
                     max_area_px=1200, max_radius_px=30)


@dataclass
class DetectionParams:
    ball_color: str = "orange"          # "white" | "orange"
    # HSV range (OpenCV conventions: H 0-179, S/V 0-255)
    hue_low: int = 5
    hue_high: int = 25
    sat_low: int = 90
    sat_high: int = 255
    val_low: int = 120
    val_high: int = 255
    # Contour filters
    min_area_px: int = 6
    max_area_px: int = 1200
    min_circularity: float = 0.35       # 4*pi*A/P^2
    min_radius_px: float = 1.5
    max_radius_px: float = 30.0
    # Motion evidence
    use_motion: bool = True
    motion_source: str = "diff"         # "diff" | "mog2" | "none"
    diff_threshold: int = 22
    min_motion_overlap: float = 0.0     # hard gate: 0 disables; 0.1+ rejects
                                        # static look-alikes (lines, shoes)
    # Morphology kernel sizes (0 disables)
    morph_open_size: int = 2
    morph_close_size: int = 3
    # Candidate scoring weights; each term is normalised to [0, 1]
    w_circularity: float = 1.0
    w_solidity: float = 0.7
    w_motion: float = 0.9
    w_distance: float = 1.6
    gating_sigma_px: float = 45.0       # Gaussian falloff towards prediction

    def apply_preset(self) -> None:
        preset = WHITE_PRESET if self.ball_color == "white" else ORANGE_PRESET
        for key, value in preset.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    """One blob that passed all appearance filters, with its score."""
    x: float
    y: float
    radius: float
    area: float
    circularity: float
    solidity: float
    motion_overlap: float
    score: float = 0.0


class BallDetector:
    def __init__(self, params: DetectionParams):
        self.params = params
        self._prev_gray: Optional[np.ndarray] = None
        self._mog2 = (cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)
            if params.motion_source == "mog2" else None)

    def reset(self) -> None:
        self._prev_gray = None
        if self._mog2 is not None:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=25, detectShadows=False)

    # ---------------------------------------------------------------- masks
    def colour_mask(self, hsv: np.ndarray) -> np.ndarray:
        p = self.params
        lo = np.array([p.hue_low, p.sat_low, p.val_low], dtype=np.uint8)
        hi = np.array([p.hue_high, p.sat_high, p.val_high], dtype=np.uint8)
        return cv2.inRange(hsv, lo, hi)

    def motion_mask(self, gray: np.ndarray) -> Optional[np.ndarray]:
        if not self.params.use_motion or self.params.motion_source == "none":
            return None
        if self._mog2 is not None:
            return self._mog2.apply(gray)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return None
        diff = cv2.absdiff(gray, self._prev_gray)
        _, mask = cv2.threshold(diff, self.params.diff_threshold, 255,
                                cv2.THRESH_BINARY)
        return cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    # ------------------------------------------------------------- pipeline
    def detect(self, frame_bgr: np.ndarray,
               predicted_xy: Optional[Tuple[float, float]] = None,
               ) -> List[Candidate]:
        p = self.params
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = self.colour_mask(hsv)
        if p.morph_open_size > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (p.morph_open_size, p.morph_open_size)))
        if p.morph_close_size > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (p.morph_close_size, p.morph_close_size)))

        motion = self.motion_mask(gray)
        self._prev_gray = gray

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Candidate] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < p.min_area_px or area > p.max_area_px:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < p.min_circularity:
                continue
            radius = float(np.sqrt(area / np.pi))
            if radius < p.min_radius_px or radius > p.max_radius_px:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity = min(area / hull_area, 1.0) if hull_area > 0 else 0.0

            m = cv2.moments(cnt)
            cx = m["m10"] / m["m00"] if m["m00"] else float(cnt[0][0][0])
            cy = m["m01"] / m["m00"] if m["m00"] else float(cnt[0][0][1])

            overlap = 0.0
            if motion is not None:
                x, y, w, h = cv2.boundingRect(cnt)
                roi_motion = motion[y:y + h, x:x + w]
                roi_mask = mask[y:y + h, x:x + w]
                inside = int(np.count_nonzero(roi_mask))
                if inside > 0:
                    overlap = float(np.count_nonzero(
                        cv2.bitwise_and(roi_mask, roi_motion))) / inside
                if p.min_motion_overlap > 0 and overlap < p.min_motion_overlap:
                    continue  # static look-alike: not moving, not the ball

            candidates.append(Candidate(cx, cy, radius, area, circularity,
                                        solidity, overlap))

        candidates = self._score(candidates, predicted_xy)
        return candidates

    def _score(self, candidates: List[Candidate],
               predicted_xy: Optional[Tuple[float, float]]) -> List[Candidate]:
        p = self.params
        use_pred = predicted_xy is not None
        total_w = (p.w_circularity + p.w_solidity + p.w_motion
                   + (p.w_distance if use_pred else 0.0))
        if total_w <= 0:
            return candidates
        for c in candidates:
            s = (p.w_circularity * float(np.clip(c.circularity / 0.9, 0, 1))
                 + p.w_solidity * c.solidity
                 + p.w_motion * c.motion_overlap)
            if use_pred:
                dx, dy = c.x - predicted_xy[0], c.y - predicted_xy[1]
                dist = float(np.hypot(dx, dy))
                s += p.w_distance * float(np.exp(
                    -0.5 * (dist / max(p.gating_sigma_px, 1.0)) ** 2))
            c.score = s / total_w
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ---------------------------------------------------------------- debug
    def debug_frame(self, frame_bgr: np.ndarray
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (colour_mask, motion_mask) for UI tuning feedback."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        colour = self.colour_mask(hsv)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        motion = self.motion_mask(gray)
        self._prev_gray = gray
        return colour, motion
