"""Output generation: annotated video, CSV tables and JSON summary."""
from __future__ import annotations

import json
import math
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .analysis import AnalysisResult, compute_speeds, project_records
from .bounce import BounceEvent
from .calibration import TableCalibration
from .tracking import TrackPoint

# BGR colours
C_MEASURED = (60, 180, 75)      # green: measured position
C_PREDICTED = (30, 160, 255)    # orange: Kalman coasted position
C_BOUNCE = (50, 60, 230)        # red: estimated bounce
C_TABLE = (200, 200, 200)
C_TEXT_BG = (25, 25, 25)

LIMITATIONS = [
    "Single fixed camera, classical image processing; no 3D reconstruction.",
    "Speeds are approximate table-plane projected speeds: the homography is "
    "exact only on the table surface, while an airborne ball lies off that "
    "plane, so its projection (and speed) is distorted by height.",
    "Bounce events are heuristic estimates from vertical motion reversal "
    "near the table region; confidence is a heuristic score, not a "
    "calibrated probability. Racket hits, net cord deflections, occlusion "
    "and unusual camera angles can produce false positives or misses.",
    "Not a professional sports measurement system and not an automatic "
    "fault/service umpiring system.",
]


def pick_writer(path: str, width: int, height: int, fps: float
                ) -> Tuple[cv2.VideoWriter, str]:
    """Open an .mp4 writer with the most browser-compatible codec available."""
    for fourcc in ("avc1", "H264", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                            fps, (width, height))
        if w.isOpened():
            return w, fourcc
        w.release()
    raise RuntimeError("No working OpenCV video writer backend for MP4.")


def _label(frame, text, org, scale=0.45, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                  thickness)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - th - 6), (x + tw + 3, y + 4),
                  C_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (245, 245, 245), thickness, cv2.LINE_AA)


def draw_table_overlay(frame: np.ndarray, cal: TableCalibration) -> None:
    quad = cal.table_quad_image()
    t = max(1, frame.shape[1] // 1280)
    cv2.polylines(frame, [quad], True, C_TABLE, t, cv2.LINE_AA)


def draw_annotation(frame: np.ndarray, rec: TrackPoint,
                    trail: List[Tuple[float, float, str]],
                    bounces_here: List[Tuple[BounceEvent, int]],
                    speed_text: str = "") -> None:
    # All overlay sizes scale with frame width so 1080p/4K videos stay legible.
    k = max(0.5, frame.shape[1] / 1920.0)
    ts = 0.45 * k          # label font scale
    lw = max(2, int(round(2 * k)))

    for tx, ty, st in trail:
        if math.isfinite(tx) and math.isfinite(ty):
            colour = C_MEASURED if st == "detected" else C_PREDICTED
            cv2.circle(frame, (int(tx), int(ty)), max(2, int(2.5 * k)),
                       colour, -1, cv2.LINE_AA)
    if len(trail) >= 2:
        pts = np.array([[(int(x), int(y)) for x, y, _ in trail
                         if math.isfinite(x) and math.isfinite(y)]], np.int32)
        cv2.polylines(frame, pts, False, (90, 90, 90), 1, cv2.LINE_AA)

    if rec.status == "detected":
        cv2.circle(frame, (int(rec.x), int(rec.y)),
                   max(int(rec.radius) + 4, int(8 * k)), C_MEASURED, lw,
                   cv2.LINE_AA)
        cv2.circle(frame, (int(rec.meas_x), int(rec.meas_y)),
                   max(2, int(2.5 * k)), C_MEASURED, -1, cv2.LINE_AA)
    elif rec.status == "predicted":
        cv2.circle(frame, (int(rec.x), int(rec.y)), max(9, int(9 * k)),
                   C_PREDICTED, lw, cv2.LINE_AA)

    for ev, age in bounces_here:
        r = int((10 + min(age, 12)) * k)
        cv2.circle(frame, (int(ev.img_x), int(ev.img_y)), r, C_BOUNCE, lw,
                   cv2.LINE_AA)
        cv2.drawMarker(frame, (int(ev.img_x), int(ev.img_y)), C_BOUNCE,
                       cv2.MARKER_CROSS, max(8, int(8 * k)), lw)
        _label(frame, f"B conf={ev.confidence:.2f}"
               + ("" if ev.certain else " (uncertain)"),
               (int(ev.img_x) + int(12 * k), int(ev.img_y) - int(10 * k)),
               0.4 * k, max(1, int(k)))

    _label(frame, f"f {rec.frame}  t {rec.t:.2f}s  {rec.status}",
           (10, int(24 * k)), ts, max(1, int(k)))
    if speed_text:
        _label(frame, speed_text, (10, frame.shape[0] - int(12 * k)), ts,
               max(1, int(k)))


def annotate_video(src_path: str, out_path: str, result: AnalysisResult,
                   speed_unit: str = "m/s (table-plane projected)",
                   progress_cb=None, trail_len: int = 30,
                   bounce_ring_frames: int = 14) -> str:
    """Second streaming pass: re-reads the source and writes the annotated MP4."""
    speeds, speeds_px, _ = compute_speeds(result.records, result.calibration,
                                          result.speed_params)
    rec_by_frame = {r.frame: (r, i) for i, r in enumerate(result.records)}
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"Could not reopen video: {src_path}")
    writer = None
    frame_index = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame is None:
                continue
            if writer is None:
                writer, fourcc = pick_writer(out_path, frame.shape[1],
                                             frame.shape[0], result.video_info.fps)
            item = rec_by_frame.get(frame_index)
            if item is None:
                writer.write(frame)
                continue
            rec, idx = item
            trail = [(r.x, r.y, r.status) for r in
                     result.records[max(0, idx - trail_len):idx + 1]
                     if r.status != "lost" and math.isfinite(r.x)]
            bounces_here = [(ev, frame_index - ev.frame) for ev in
                            result.bounces
                            if 0 <= frame_index - ev.frame <= bounce_ring_frames]
            sp = speeds[idx] if math.isfinite(speeds[idx]) else math.nan
            stext = ""
            if math.isfinite(sp):
                if result.calibration is not None:
                    stext = f"{sp:.1f} m/s | {sp * 3.6:.1f} km/h (approx. table-plane projected)"
                else:
                    stext = f"{sp:.0f} px/s (uncalibrated)"
            if result.calibration is not None:
                draw_table_overlay(frame, result.calibration)
            draw_annotation(frame, rec, trail, bounces_here, stext)
            writer.write(frame)
            if progress_cb and frame_index % 20 == 0:
                progress_cb(frame_index + 1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return out_path


def per_frame_dataframe(result: AnalysisResult) -> pd.DataFrame:
    speeds, speeds_px, outliers = compute_speeds(
        result.records, result.calibration, result.speed_params)
    tx, ty = project_records(result.records, result.calibration)
    rows = []
    for i, r in enumerate(result.records):
        rows.append({
            "frame": r.frame,
            "timestamp_s": round(r.t, 4),
            "timestamp_source": result.video_info.timestamp_source,
            "status": r.status,
            "measured_x_px": None if not math.isfinite(r.meas_x) else round(r.meas_x, 1),
            "measured_y_px": None if not math.isfinite(r.meas_y) else round(r.meas_y, 1),
            "measured_radius_px": None if not math.isfinite(r.radius) else round(r.radius, 1),
            "detection_score": None if not math.isfinite(r.score) else round(r.score, 3),
            "filter_x_px": None if not math.isfinite(r.x) else round(r.x, 1),
            "filter_y_px": None if not math.isfinite(r.y) else round(r.y, 1),
            "filter_vx_px_s": None if not math.isfinite(r.vx) else round(r.vx, 1),
            "filter_vy_px_s": None if not math.isfinite(r.vy) else round(r.vy, 1),
            "table_x_m": None if not math.isfinite(tx[i]) else round(tx[i], 3),
            "table_y_m": None if not math.isfinite(ty[i]) else round(ty[i], 3),
            "speed_table_plane_mps": None if not math.isfinite(speeds[i]) else round(speeds[i], 3),
            "speed_table_plane_kmh": None if not math.isfinite(speeds[i]) else round(speeds[i] * 3.6, 2),
            "speed_pxps": None if not math.isfinite(speeds_px[i]) else round(speeds_px[i], 1),
            "speed_outlier": bool(outliers[i]),
        })
    return pd.DataFrame(rows)


def bounces_dataframe(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_index": i + 1,
        "frame": ev.frame,
        "timestamp_s": round(ev.t, 4),
        "image_x_px": round(ev.img_x, 1),
        "image_y_px": round(ev.img_y, 1),
        "table_x_m": None if not math.isfinite(ev.table_x) else round(ev.table_x, 3),
        "table_y_m": None if not math.isfinite(ev.table_y) else round(ev.table_y, 3),
        "vy_before_px_per_frame": round(ev.vy_before, 2),
        "vy_after_px_per_frame": round(ev.vy_after, 2),
        "confidence_heuristic": round(ev.confidence, 3),
        "certainty": "certain" if ev.certain else "uncertain",
    } for i, ev in enumerate(result.bounces)])


def summary_json(result: AnalysisResult, writer_fourcc: Optional[str] = None,
                 extra: Optional[dict] = None) -> str:
    payload = {
        "project": "Vision-Based Table Tennis Ball Tracking, Bounce Detection "
                   "and Speed Estimation (college Image & Video Processing)",
        "video": {
            "path_name": Path(result.video_info.path).name,
            "width_px": result.video_info.width,
            "height_px": result.video_info.height,
            "fps": result.video_info.fps,
            "fps_is_default_assumption": result.video_info.fps_is_default,
            "frame_count_header": result.video_info.frame_count,
            "timestamp_source": result.video_info.timestamp_source,
            "warnings": result.video_info.warnings,
        },
        "calibration": (result.calibration.to_dict()
                        if result.calibration else None),
        "settings": {
            "detection": result.detection_params.to_dict(),
            "tracking": result.tracking_params.to_dict(),
            "bounce": result.bounce_params.to_dict(),
            "speed": dict(result.speed_params.__dict__),
        },
        "summary": result.stats,
        "annotated_video": {"codec_fourcc": writer_fourcc},
        "measurement_limitations": LIMITATIONS,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2)


def new_artifact_dir(prefix: str = "ttv") -> Path:
    base = Path(tempfile.gettempdir()) / "ttv_app"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d
