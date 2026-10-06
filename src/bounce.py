"""Heuristic bounce detection from the measured trajectory.

Method (deliberately transparent, no learned model):
  1. Smooth short windows of the MEASURED y positions (image coords, y down).
  2. Around each measured frame, fit the vertical velocity before and after
     the candidate frame over configurable windows.
  3. A bounce candidate is a downward-to-upward vertical reversal whose
     combined strength exceeds a threshold, located near the table region.
  4. Require enough measured detections around the event and reject events
     next to long tracking gaps.
  5. Apply a cooldown so one physical bounce is not counted twice.

A vertical reversal alone is NOT proof of a bounce: a racket hit, a net cord
deflection, occlusion artefacts or an unusual camera angle can all cause one.
The confidence score is a heuristic (0-1) combining proximity to the table,
reversal strength and measured coverage — it is NOT a calibrated probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .calibration import TableCalibration
from .tracking import TrackPoint


@dataclass
class BounceParams:
    smooth_window: int = 5          # samples (measured points) centred mean
    velocity_window: int = 6        # frames each side for vy estimation
    min_reversal_strength: float = 2.0   # px/frame, |vy_before|+|vy_after|
    min_measured_around: int = 4    # measured frames within +-window
    max_gap_frames: int = 4         # max frame gap near the event
    cooldown_frames: int = 10       # min separation between distinct events
    table_margin_m: float = 0.35    # tolerance for "near the table plane"
    min_confidence: float = 0.30
    uncertain_threshold: float = 0.55

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class BounceEvent:
    frame: int
    t: float
    img_x: float
    img_y: float
    table_x: float = math.nan
    table_y: float = math.nan
    confidence: float = 0.0
    certain: bool = False
    vy_before: float = 0.0
    vy_after: float = 0.0


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    kernel = np.ones(int(window)) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _slope(frames: np.ndarray, times: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares dy/dt; returns 0.0 when undetermined."""
    if len(frames) < 2:
        return 0.0
    t_span = times[-1] - times[0]
    if t_span <= 1e-9:
        return 0.0
    return float(np.polyfit(times, ys, 1)[0])  # px per second


def detect_bounces(records: List[TrackPoint],
                   calibration: Optional[TableCalibration] = None,
                   params: Optional[BounceParams] = None) -> List[BounceEvent]:
    p = params or BounceParams()
    measured = [(r.frame, r.t, r.meas_x, r.meas_y) for r in records
                if r.status == "detected" and np.isfinite(r.meas_y)]
    if len(measured) < max(4, 2 * p.velocity_window):
        return []

    frames = np.array([m[0] for m in measured], dtype=float)
    times = np.array([m[1] for m in measured], dtype=float)
    xs = np.array([m[2] for m in measured], dtype=float)
    ys = _moving_average(np.array([m[3] for m in measured], dtype=float),
                         p.smooth_window)

    fps = _median_rate(frames, times)
    events: List[BounceEvent] = []
    for i in range(1, len(frames) - 1):
        f = frames[i]
        pre_idx = np.where((frames < f) & (frames >= f - p.velocity_window))[0]
        post_idx = np.where((frames > f) & (frames <= f + p.velocity_window))[0]
        if len(pre_idx) < 2 or len(post_idx) < 2:
            continue

        # Reject candidates adjacent to long tracking gaps.
        local = np.concatenate([[f], frames[pre_idx], frames[post_idx]])
        local.sort()
        if np.any(np.diff(local) > p.max_gap_frames):
            continue

        coverage = (len(pre_idx) + len(post_idx)) / float(2 * p.velocity_window)
        if len(pre_idx) + len(post_idx) < p.min_measured_around:
            continue

        # px/frame so thresholds are frame-rate independent.
        vy_before = _slope(frames[pre_idx], times[pre_idx], ys[pre_idx]) / fps
        vy_after = _slope(frames[post_idx], times[post_idx], ys[post_idx]) / fps
        if not (vy_before > 0.0 and vy_after < 0.0):
            continue
        strength = abs(vy_before) + abs(vy_after)
        if strength < p.min_reversal_strength:
            continue

        prox = 0.4  # uncalibrated: cannot verify table proximity
        tx = ty = math.nan
        if calibration is not None:
            tx, ty = calibration.project(xs[i], ys[i])
            if calibration.inside_table(tx, ty, p.table_margin_m):
                prox = 1.0
            elif calibration.inside_table(tx, ty, 2.0 * p.table_margin_m):
                prox = 0.5
            else:
                continue  # reversal far from the table plane: not a bounce

        strength_term = float(np.clip(strength / (2.0 * p.min_reversal_strength
                                                  * 2.0), 0.0, 1.0))
        confidence = float(np.clip(
            0.35 * prox + 0.30 * strength_term + 0.35 * coverage, 0.0, 1.0))
        if confidence < p.min_confidence:
            continue

        events.append(BounceEvent(
            frame=int(f), t=float(times[i]), img_x=float(xs[i]),
            img_y=float(ys[i]), table_x=tx, table_y=ty,
            confidence=confidence, certain=confidence >= p.uncertain_threshold,
            vy_before=float(vy_before), vy_after=float(vy_after)))

    # Cooldown / duplicate suppression: keep the stronger event in a cluster.
    events.sort(key=lambda e: (e.frame, -e.confidence))
    accepted: List[BounceEvent] = []
    for e in events:
        if accepted and e.frame - accepted[-1].frame < p.cooldown_frames:
            if e.confidence > accepted[-1].confidence:
                accepted[-1] = e
            continue
        accepted.append(e)
    return accepted


def _median_rate(frames: np.ndarray, times: np.ndarray) -> float:
    dts = np.diff(times)
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        dts = np.diff(frames)
    rate = 1.0 / max(float(np.median(dts)), 1e-6)
    return rate
