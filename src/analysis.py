"""Streaming video analysis pipeline and honest speed estimation.

Pass 1 (this module) streams frames from disk, runs detection + tracking and
records per-frame results.  Bounces and speeds are computed from MEASURED
points only; Kalman coasted ("predicted") points keep the visual track
continuous but are never used as evidence for speed or bounce confirmation.

Timestamps come from the container (POS_MSEC) when the probe finds them
monotonic; otherwise frame_index / video_fps.  Processing speed never enters
the math.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np
import pandas as pd

from .bounce import BounceEvent, BounceParams, detect_bounces
from .calibration import TableCalibration
from .detection import BallDetector, DetectionParams
from .tracking import BallTracker, TrackPoint, TrackingParams


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int          # 0 when the header does not report it
    duration_s: float
    timestamp_source: str     # "container-msec" | "frame-index/fps"
    fps_is_default: bool
    frame_count_is_estimate: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_is_known(self) -> bool:
        return self.frame_count > 0


def probe_video(path: str, sample: int = 25) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open the video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps_is_default = not (fps > 1.0)
        if fps_is_default:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        warnings: List[str] = []
        if fps_is_default:
            warnings.append(
                "Video FPS metadata missing/invalid; assuming 30 fps. "
                "Timestamps and speeds may be scaled incorrectly.")
        if width <= 0 or height <= 0:
            raise IOError("Video reports an invalid frame size.")

        msecs, read = [], 0
        while read < sample:
            ok, frame = cap.read()
            if not ok:
                break
            msecs.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
            read += 1
        monotonic = (len(msecs) >= 5
                     and all(b > a for a, b in zip(msecs, msecs[1:])))
        if monotonic:
            rates = np.diff(msecs)
            if not (0.5 * 1000.0 / fps <= float(np.median(rates))
                    <= 2.0 * 1000.0 / fps):
                monotonic = False
        timestamp_source = "container-msec" if monotonic else "frame-index/fps"
        if not monotonic and len(msecs) >= 5:
            warnings.append(
                "Container timestamps look unreliable; using frame index / "
                "FPS for time instead.")

        duration = frame_count / fps if frame_count > 0 else (read / fps)
        return VideoInfo(path=path, width=width, height=height, fps=round(fps, 3),
                         frame_count=frame_count, duration_s=duration,
                         timestamp_source=timestamp_source,
                         fps_is_default=fps_is_default,
                         frame_count_is_estimate=frame_count == 0,
                         warnings=warnings)
    finally:
        cap.release()


@dataclass
class SpeedParams:
    max_gap_frames: int = 4      # no speed across gaps larger than this
    outlier_mad_z: float = 3.5   # robust z-score cut for outlier rejection
    smooth_window: int = 5       # centred rolling median (modest smoothing)


@dataclass
class AnalysisResult:
    records: List[TrackPoint]
    bounces: List[BounceEvent]
    video_info: VideoInfo
    calibration: Optional[TableCalibration]
    stats: dict
    speed_params: SpeedParams
    detection_params: DetectionParams
    tracking_params: TrackingParams
    bounce_params: BounceParams
    processed_frames: int = 0

    def measured_coverage(self) -> float:
        if not self.processed_frames:
            return 0.0
        n = sum(1 for r in self.records if r.status == "detected")
        return n / float(self.processed_frames)


def _reject_outliers_hampel(values: np.ndarray, window: int = 7,
                            z: float = 3.5) -> np.ndarray:
    """Leave-one-out Hampel filter: each value is compared against the
    median/MAD of its neighbourhood with itself excluded.  Falls back to a
    relative floor when the neighbourhood MAD is zero (perfectly constant
    speeds), which plain MAD cannot handle."""
    out = np.zeros(values.shape, dtype=bool)
    finite_idx = np.where(np.isfinite(values))[0]
    half = max(window // 2, 1)
    for i in finite_idx:
        neigh = values[max(0, i - half):min(len(values), i + half + 1)]
        neigh = neigh[np.isfinite(neigh)]
        neigh = neigh[neigh != values[i]] if (neigh == values[i]).any() else neigh
        if neigh.size < 3:
            continue
        med = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - med)))
        if mad > 1e-9:
            thresh = z * mad / 0.6745
        else:
            # Discrete/quantised speeds make MAD collapse to zero: when the
            # majority of the neighbourhood sits exactly at the median, any
            # value far from it is treated as an outlier.
            thresh = 0.5 * abs(med) + 1e-9
        if abs(values[i] - med) > thresh:
            out[i] = True
    return out


def compute_speeds(records: List[TrackPoint],
                   calibration: Optional[TableCalibration],
                   params: Optional[SpeedParams] = None):
    """Per-frame speeds from consecutive MEASURED positions.

    Returns (speed_primary, speed_pxps, outliers_mask) aligned with `records`.
    speed_primary is m/s when calibrated ("approximate table-plane projected
    speed") and px/s otherwise.
    """
    p = params or SpeedParams()
    n = len(records)
    speeds = np.full(n, np.nan)
    speeds_px = np.full(n, np.nan)
    measured_idx = [i for i, r in enumerate(records)
                    if r.status == "detected" and np.isfinite(r.meas_x)
                    and np.isfinite(r.meas_y) and np.isfinite(r.t)]
    for a_i, b_i in zip(measured_idx, measured_idx[1:]):
        a, b = records[a_i], records[b_i]
        gap = b.frame - a.frame
        if gap < 1 or gap > p.max_gap_frames:
            continue
        dt = b.t - a.t
        if dt <= 1e-6:
            continue
        dpx = math.hypot(b.meas_x - a.meas_x, b.meas_y - a.meas_y)
        speeds_px[b_i] = dpx / dt
        if calibration is not None:
            ax, ay = calibration.project(a.meas_x, a.meas_y)
            bx, by = calibration.project(b.meas_x, b.meas_y)
            speeds[b_i] = math.hypot(bx - ax, by - ay) / dt

    primary = speeds if calibration is not None else speeds_px
    outliers = _reject_outliers_hampel(primary, window=p.smooth_window + 2,
                                       z=p.outlier_mad_z)
    primary = primary.copy()
    primary[outliers] = np.nan

    if p.smooth_window > 1 and np.any(np.isfinite(primary)):
        ser = pd.Series(primary).rolling(
            p.smooth_window, center=True, min_periods=1).median()
        smoothed = ser.to_numpy()
        keep = np.isfinite(primary)  # never invent values for missing frames
        primary[keep] = smoothed[keep]
    if calibration is None:
        speeds_px = primary
    else:
        speeds = primary
    return speeds, speeds_px, outliers


def project_records(records: List[TrackPoint],
                    calibration: Optional[TableCalibration]):
    """Table-plane coordinates for measured points only (NaN elsewhere)."""
    tx = np.full(len(records), np.nan)
    ty = np.full(len(records), np.nan)
    if calibration is None:
        return tx, ty
    idx = [i for i, r in enumerate(records)
           if r.status == "detected" and np.isfinite(r.meas_x)]
    if not idx:
        return tx, ty
    xy = np.array([[records[i].meas_x, records[i].meas_y] for i in idx])
    xs, ys = calibration.project_many(xy)
    for k, i in enumerate(idx):
        tx[i], ty[i] = xs[k], ys[k]
    return tx, ty


def summarise(records, bounces, speeds, speeds_px, calibration, video_info,
              processed_frames, duration_s) -> dict:
    def series_stats(a):
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            return math.nan, math.nan
        return float(np.median(fin)), float(np.max(fin))

    med_m, max_m = series_stats(speeds)
    med_px, max_px = series_stats(speeds_px)
    n_measured = sum(1 for r in records if r.status == "detected")
    n_predicted = sum(1 for r in records if r.status == "predicted")
    return {
        "duration_s": duration_s,
        "processed_frames": processed_frames,
        "measured_frames": n_measured,
        "predicted_frames": n_predicted,
        "measured_coverage": n_measured / processed_frames if processed_frames else 0.0,
        "estimated_bounces_total": len(bounces),
        "estimated_bounces_certain": sum(1 for b in bounces if b.certain),
        "estimated_bounces_uncertain": sum(1 for b in bounces if not b.certain),
        "speed_unit": "m/s (table-plane projected)" if calibration else "px/s",
        "median_projected_speed_mps": med_m,
        "max_projected_speed_mps": max_m,
        "median_pixel_speed_pxps": med_px,
        "max_pixel_speed_pxps": max_px,
    }


class VideoAnalyzer:
    def __init__(self, video_path: str,
                 detection: Optional[DetectionParams] = None,
                 tracking: Optional[TrackingParams] = None,
                 bounce: Optional[BounceParams] = None,
                 speed: Optional[SpeedParams] = None,
                 calibration: Optional[TableCalibration] = None,
                 video_info: Optional[VideoInfo] = None):
        self.path = video_path
        self.det_params = detection or DetectionParams()
        self.trk_params = tracking or TrackingParams()
        self.bounce_params = bounce or BounceParams()
        self.speed_params = speed or SpeedParams()
        self.calibration = calibration
        self.info = video_info or probe_video(video_path)

    def _timestamp(self, cap: cv2.VideoCapture, frame_index: int) -> float:
        if self.info.timestamp_source == "container-msec":
            return float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        return frame_index / self.info.fps

    def analyze(self, start_frame: int = 0, max_frames: Optional[int] = None,
                progress_cb: Optional[Callable[[float], None]] = None
                ) -> AnalysisResult:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"Could not reopen video: {self.path}")
        detector = BallDetector(self.det_params)
        tracker = BallTracker(self.trk_params, self.info.fps)
        records: List[TrackPoint] = []
        frame_index = -1
        processed = 0
        prev_t: Optional[float] = None
        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_index = start_frame - 1
            budget = max_frames if max_frames is not None else float("inf")
            while processed < budget:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                t = self._timestamp(cap, frame_index)
                if prev_t is not None and t <= prev_t:
                    t = prev_t + 1.0 / self.info.fps  # keep time monotonic
                dt = (t - prev_t) if prev_t is not None else 1.0 / self.info.fps
                prev_t = t

                predicted = tracker.kf.pos if tracker.is_tracking else None
                candidates = detector.detect(frame, predicted)
                records.append(tracker.step(candidates, dt, frame_index, t))
                processed += 1
                if progress_cb and processed % 10 == 0:
                    progress_cb(processed)
        finally:
            cap.release()

        bounces = detect_bounces(records, self.calibration, self.bounce_params)
        speeds, speeds_px, _ = compute_speeds(
            records, self.calibration, self.speed_params)
        duration = processed / self.info.fps if processed else 0.0
        if records:
            duration = max(duration, records[-1].t - records[0].t
                           + 1.0 / self.info.fps)
        stats = summarise(records, bounces, speeds, speeds_px, self.calibration,
                          self.info, processed, duration)
        return AnalysisResult(records=records, bounces=bounces,
                              video_info=self.info, calibration=self.calibration,
                              stats=stats, speed_params=self.speed_params,
                              detection_params=self.det_params,
                              tracking_params=self.trk_params,
                              bounce_params=self.bounce_params,
                              processed_frames=processed)
