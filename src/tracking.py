"""Kalman-filter based single-ball tracking with controlled reacquisition.

State: [x, y, vx, vy] (pixels, pixels/second), constant-velocity model.
Measurements: [x, y] from the detector.  The transition matrix is rebuilt
with the actual frame-to-frame dt on every predict call, so prediction is
frame-rate aware (uses video timestamps, not the processing speed).

Per-frame status:
  - "detected":  a gated candidate was matched and used to correct the filter.
  - "predicted": no candidate matched; the coasted Kalman state is reported.
                 Predicted points keep the visual track continuous but are
                 NOT treated as measured evidence for speed or bounces.
  - "lost":      tracking reset (missed-frame limit exceeded or no track).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TrackingParams:
    max_missed_frames: int = 12      # coast this long, then reset tracking
    reacq_confirm: int = 2           # consecutive detections before re-lock
    max_jump_px: float = 200.0       # gating radius around the prediction
    measurement_noise_px: float = 3.0
    process_noise: float = 2500.0    # (px/s^2)^2; ball accelerates sharply
    initial_vel_var: float = 1.0e6  # (px/s)^2; unknown velocity at lock-on

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TrackPoint:
    frame: int
    t: float
    status: str                 # "detected" | "predicted" | "lost"
    meas_x: float = math.nan
    meas_y: float = math.nan
    radius: float = math.nan
    score: float = math.nan     # detector score of the matched candidate
    x: float = math.nan         # reported (filter) position
    y: float = math.nan
    vx: float = math.nan
    vy: float = math.nan


class KalmanCV:
    """Constant-velocity Kalman filter over pixel position and velocity."""

    def __init__(self, x: float, y: float, process_noise: float = 2500.0,
                 measurement_noise_px: float = 3.0):
        self._process_noise = float(process_noise)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(
            measurement_noise_px ** 2)
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        cov = np.eye(4, dtype=np.float32)
        cov[2, 2] = cov[3, 3] = 1.0e6
        self.kf.errorCovPost = cov

    def predict(self, dt: float) -> np.ndarray:
        f = np.eye(4, dtype=np.float32)
        f[0, 2] = f[1, 3] = dt
        self.kf.transitionMatrix = f
        q = np.zeros((4, 4), dtype=np.float32)
        q[0, 0] = q[1, 1] = 0.25 * dt ** 4
        q[0, 2] = q[2, 0] = q[1, 3] = q[3, 1] = 0.5 * dt ** 3
        q[2, 2] = q[3, 3] = dt ** 2
        self.kf.processNoiseCov = q * self._process_noise
        return self.kf.predict()

    def correct(self, x: float, y: float) -> None:
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))

    @property
    def pos(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])

    @property
    def vel(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])


class BallTracker:
    def __init__(self, params: TrackingParams, fps: float):
        self.params = params
        self.fps = max(float(fps), 1e-3)
        self.kf: Optional[KalmanCV] = None
        self.miss_streak = 0
        self.confirm_count = 0
        self._pending: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.kf = None
        self.miss_streak = 0
        self.confirm_count = 0
        self._pending = None

    @property
    def is_tracking(self) -> bool:
        return self.kf is not None

    def _reacquisition_ok(self, cxy: Tuple[float, float]) -> bool:
        """Controlled reacquisition: candidate must stay near the pending
        position, with a tolerance that grows with elapsed frames."""
        if self._pending is None:
            return True
        px, py = self._pending
        tol = self.params.max_jump_px * max(1, self.confirm_count)
        return math.hypot(cxy[0] - px, cxy[1] - py) <= tol

    def step(self, candidates: List, dt: float, frame: int, t: float
             ) -> TrackPoint:
        dt = float(np.clip(dt, 1.0 / (30 * 30), 0.5))

        if not self.is_tracking:
            return self._step_search(candidates, dt, frame, t)

        self.kf.predict(dt)
        gated = [c for c in candidates
                 if math.hypot(c.x - self.kf.pos[0], c.y - self.kf.pos[1])
                 <= self.params.max_jump_px]
        if gated:
            best = max(gated, key=lambda c: c.score)
            self.kf.correct(best.x, best.y)
            self.miss_streak = 0
            px, py = self.kf.pos
            vx, vy = self.kf.vel
            return TrackPoint(frame, t, "detected", best.x, best.y,
                              best.radius, best.score, px, py, vx, vy)

        self.miss_streak += 1
        if self.miss_streak > self.params.max_missed_frames:
            self.reset()
            return TrackPoint(frame, t, "lost")
        px, py = self.kf.pos
        vx, vy = self.kf.vel
        return TrackPoint(frame, t, "predicted", x=px, y=py, vx=vx, vy=vy)

    def _step_search(self, candidates: List, dt: float, frame: int, t: float
                     ) -> TrackPoint:
        if not candidates:
            self.confirm_count = 0
            self._pending = None
            return TrackPoint(frame, t, "lost")

        best = candidates[0]  # already score-ranked by the detector
        if self._reacquisition_ok((best.x, best.y)):
            self._pending = (best.x, best.y)
            self.confirm_count += 1
        else:
            self.confirm_count = 1
            self._pending = (best.x, best.y)

        if self.confirm_count >= max(1, self.params.reacq_confirm):
            self.kf = KalmanCV(best.x, best.y,
                               process_noise=self.params.process_noise,
                               measurement_noise_px=self.params.measurement_noise_px)
            self.kf.predict(dt)
            self.kf.correct(best.x, best.y)
            self.miss_streak = 0
            self.confirm_count = 0
            self._pending = None
        # Reported as measured: it is a raw detector observation; the filter
        # itself is (re-)initialised only after confirmation.
        return TrackPoint(frame, t, "detected", best.x, best.y, best.radius,
                          best.score, best.x, best.y, 0.0, 0.0)
