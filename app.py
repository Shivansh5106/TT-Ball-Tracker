"""Streamlit app: Vision-Based Table Tennis Ball Tracking, Bounce Detection
and Speed Estimation.

Four numbered stages (Video -> Calibration -> Preview -> Analysis).  Heavy
processing only runs behind explicit Preview / Analyze buttons, so widget
changes rerender instantly.  Speeds are reported as *approximate table-plane
projected speeds* (pixel speeds when uncalibrated) — never as true 3D speeds.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (AnalysisResult, SpeedParams, VideoAnalyzer,
                          compute_speeds, probe_video)
from src.bounce import BounceParams
from src.calibration import (CORNER_GUIDE, CORNER_ORDER,
                             CalibrationError, TableCalibration)
from src.detection import (ORANGE_PRESET, WHITE_PRESET,
                           DetectionParams)
from src.export import (LIMITATIONS, annotate_video, bounces_dataframe,
                        draw_annotation, draw_table_overlay, new_artifact_dir,
                        per_frame_dataframe, summary_json)
from src.tracking import TrackingParams

st.set_page_config(page_title="Table Tennis Ball Tracker",
                   page_icon="🏓", layout="wide",
                   menu_items={"About": "College Image & Video Processing "
                               "project — classical ball tracking, heuristic "
                               "bounce detection, approximate table-plane "
                               "projected speeds."})

ACCENT = "#E85D26"
st.markdown(f"""
<style>
  .stApp {{ background: #FAFAF8; color: #1B1B1B; }}
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: #141414;
      letter-spacing: -0.01em; }}
  [data-testid="stHeader"] {{ background: #FAFAF8; }}

  /* Numbered stage headers */
  .stage-title {{ margin: 0.45rem 0 0.1rem 0; font-size: 1.5rem; }}
  .stage-badge {{ display:inline-block; background:{ACCENT}; color:#fff;
      font-weight:800; border-radius:9px; padding:1px 12px; margin-right:10px;
      font-size:1.0rem; vertical-align:3px; }}
  .stage-cap {{ color:#6B6660; font-size:0.88rem; margin:0 0 0.7rem 0; }}

  /* Metric cards */
  .metric-card {{ background:#fff; border:1px solid #E7E4DE; border-radius:12px;
      padding:10px 14px; height:100%; box-shadow:0 1px 2px rgba(20,20,20,.04); }}
  .metric-card .label {{ font-size:0.71rem; color:#6B6660; text-transform:uppercase;
      letter-spacing:.06em; font-weight:700; }}
  .metric-card .value {{ font-size:1.28rem; font-weight:700; color:#1B1B1B; }}
  .metric-card .sub {{ font-size:0.72rem; color:#8A857E; line-height:1.35; }}

  /* Panels */
  .panel {{ background:#fff; border:1px solid #E7E4DE; border-radius:14px;
      padding:18px 22px; }}
  .panel ol {{ margin:0.45rem 0 0.7rem 1.15rem; padding:0; }}
  .panel li {{ margin:0.18rem 0; }}

  /* Expanders */
  div[data-testid="stExpander"] {{ background:#fff; border:1px solid #E7E4DE;
      border-radius:10px; }}
  div[data-testid="stExpander"] summary {{ font-weight:600; font-size:0.9rem; }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{ background:#FFFFFF;
      border-right:1px solid #E7E4DE; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top:1.3rem; }}
  .side-brand {{ font-weight:800; font-size:1.06rem; margin:0 0 0.2rem 0; }}

  .stButton > button {{ border-radius:10px; font-weight:600; }}
  hr {{ border-color:#EDEAE4; }}
  [data-testid="stFileUploaderDropzone"] {{ background:#fff; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------- utilities
def stage_header(num: int, title: str, caption: str = "") -> None:
    st.markdown(f'<h2 class="stage-title"><span class="stage-badge">{num}</span>'
                f"{title}</h2>", unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="stage-cap">{caption}</p>',
                    unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def fmt(v: float, spec: str = ".1f") -> str:
    """NaN-safe number formatting for metric cards."""
    return format(v, spec) if isinstance(v, float) and math.isfinite(v) else "—"


def defaults() -> None:
    values = dict(
        ball_color="auto",
        det_hue_low=ORANGE_PRESET["hue_low"], det_hue_high=ORANGE_PRESET["hue_high"],
        det_sat_low=ORANGE_PRESET["sat_low"], det_sat_high=ORANGE_PRESET["sat_high"],
        det_val_low=ORANGE_PRESET["val_low"], det_val_high=ORANGE_PRESET["val_high"],
        det_min_area=DetectionParams().min_area_px,
        det_max_area=ORANGE_PRESET["max_area_px"],
        det_min_circ=ORANGE_PRESET["min_circularity"],
        det_max_radius=ORANGE_PRESET["max_radius_px"],
        det_min_motion=ORANGE_PRESET["min_motion_overlap"],
        det_use_motion=True, det_motion_source="diff",
        det_diff_threshold=DetectionParams().diff_threshold,
        trk_max_missed=TrackingParams().max_missed_frames,
        trk_max_jump=int(TrackingParams().max_jump_px),
        trk_meas_noise=TrackingParams().measurement_noise_px,
        trk_process_noise=int(TrackingParams().process_noise),
        bnc_cooldown=BounceParams().cooldown_frames,
        bnc_min_strength=BounceParams().min_reversal_strength,
        bnc_table_margin=BounceParams().table_margin_m,
        bnc_uncertain=BounceParams().uncertain_threshold,
        spd_max_gap=SpeedParams().max_gap_frames,
        spd_smooth=SpeedParams().smooth_window,
    )
    for k, v in values.items():
        st.session_state.setdefault(k, v)


# Preset dict keys -> session-state keys that don't follow the det_<key> rule.
PRESET_KEY_MAP = {"min_circularity": "det_min_circ", "max_area_px": "det_max_area",
                  "max_radius_px": "det_max_radius",
                  "min_motion_overlap": "det_min_motion"}


def apply_preset_callback() -> None:
    if st.session_state["ball_color"] == "auto":
        return  # resolved at run time from the video itself
    preset = WHITE_PRESET if st.session_state["ball_color"] == "white" else ORANGE_PRESET
    for k, v in preset.items():
        st.session_state[PRESET_KEY_MAP.get(k, f"det_{k}")] = v


def preset_params(color: str) -> DetectionParams:
    det = DetectionParams(ball_color=color)
    det.apply_preset()
    return det


def auto_pick_color(video_path: str, info, cal, trk, bnc, spd,
                    n_frames: int = 240):
    """Try both colour presets on a short segment and pick the more
    plausible ball.  The ball is the fastest object in table-tennis footage,
    so among presets that detect anything at all (coverage ≥ 5 %) the higher
    median measured speed wins; coverage breaks ties.  Returns
    (winner, {color: AnalysisResult})."""
    attempts = {}
    for color in ("orange", "white"):
        analyzer = VideoAnalyzer(video_path, detection=preset_params(color),
                                 tracking=trk, bounce=bnc, speed=spd,
                                 calibration=cal, video_info=info)
        attempts[color] = analyzer.analyze(max_frames=n_frames)

    def score(color: str):
        s = attempts[color].stats
        cov = s["measured_coverage"]
        med = s["median_pixel_speed_pxps"] or 0.0
        return (med if cov >= 0.05 else -1.0, cov)

    winner = max(("orange", "white"), key=score)
    return winner, attempts


def build_params() -> dict:
    s = st.session_state
    det = DetectionParams(
        ball_color=s.ball_color, hue_low=int(s.det_hue_low), hue_high=int(s.det_hue_high),
        sat_low=int(s.det_sat_low), sat_high=int(s.det_sat_high),
        val_low=int(s.det_val_low), val_high=int(s.det_val_high),
        min_area_px=int(s.det_min_area), max_area_px=int(s.det_max_area),
        min_circularity=float(s.det_min_circ), max_radius_px=float(s.det_max_radius),
        use_motion=bool(s.det_use_motion), motion_source=s.det_motion_source,
        diff_threshold=int(s.det_diff_threshold),
        min_motion_overlap=float(s.det_min_motion))
    trk = TrackingParams(
        max_missed_frames=int(s.trk_max_missed), max_jump_px=float(s.trk_max_jump),
        measurement_noise_px=float(s.trk_meas_noise),
        process_noise=float(s.trk_process_noise))
    bnc = BounceParams(
        cooldown_frames=int(s.bnc_cooldown),
        min_reversal_strength=float(s.bnc_min_strength),
        table_margin_m=float(s.bnc_table_margin),
        uncertain_threshold=float(s.bnc_uncertain))
    spd = SpeedParams(max_gap_frames=int(s.spd_max_gap),
                      smooth_window=int(s.spd_smooth))
    return dict(detection=det, tracking=trk, bounce=bnc, speed=spd)


def grab_frame(path: str, pos: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frame = None
    try:
        if pos > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok and pos > 0:      # seek unsupported -> first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def render_preview_frames(path: str, result: AnalysisResult,
                          count: int = 6) -> List[tuple]:
    """Returns [(frame_index, annotated RGB image), ...] sampled evenly."""
    indices = set(np.linspace(0, max(len(result.records) - 1, 0),
                              count).astype(int))
    speeds, _, _ = compute_speeds(result.records, result.calibration,
                                  result.speed_params)
    rec_by_frame = {r.frame: (r, i) for i, r in enumerate(result.records)}
    out: List[np.ndarray] = []
    cap = cv2.VideoCapture(path)
    try:
        f = -1
        while len(out) < len(indices):
            ok, frame = cap.read()
            if not ok:
                break
            f += 1
            if f not in indices or frame is None:
                continue
            item = rec_by_frame.get(f)
            if item:
                rec, idx = item
                trail = [(r.x, r.y, r.status) for r in
                         result.records[max(0, idx - 30):idx + 1]
                         if r.status != "lost" and math.isfinite(r.x)]
                here = [(ev, f - ev.frame) for ev in result.bounces
                        if 0 <= f - ev.frame <= 14]
                sp = speeds[idx]
                stext = (f"{sp:.1f} m/s | {sp*3.6:.1f} km/h (approx. table-plane projected)"
                         if math.isfinite(sp) and result.calibration
                         else (f"{sp:.0f} px/s (uncalibrated)" if math.isfinite(sp) else ""))
                if result.calibration:
                    draw_table_overlay(frame, result.calibration)
                draw_annotation(frame, rec, trail, here, stext)
            out.append((f, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return out


def colour_mask_preview(frame_bgr: np.ndarray, det: DetectionParams) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([det.hue_low, det.sat_low, det.val_low], np.uint8)
    hi = np.array([det.hue_high, det.sat_high, det.val_high], np.uint8)
    return cv2.inRange(hsv, lo, hi)


# --------------------------------------------------------- sidebar settings
defaults()
with st.sidebar:
    st.markdown('<p class="side-brand">🏓 Table Tennis Ball Tracker</p>',
                unsafe_allow_html=True)
    st.caption("Classical image & video processing project — fixed camera, "
               "recorded video. Speeds are approximate table-plane "
               "projections, not true 3D speeds.")
    st.selectbox("Ball colour", ["auto", "orange", "white"], key="ball_color",
                 on_change=apply_preset_callback,
                 help="Auto tries both colour presets on the preview segment "
                      "and keeps the one tracking the faster, more consistent "
                      "object. Pick a colour to use the sliders below.")
    with st.expander("Advanced · detection tuning"):
        if st.session_state["ball_color"] == "auto":
            st.caption("Ball colour is **auto** — these sliders apply only "
                       "when you pick orange or white explicitly.")
        st.slider("Hue low", 0, 179, key="det_hue_low")
        st.slider("Hue high", 0, 179, key="det_hue_high")
        st.slider("Saturation low", 0, 255, key="det_sat_low")
        st.slider("Saturation high", 0, 255, key="det_sat_high")
        st.slider("Value low", 0, 255, key="det_val_low")
        st.slider("Value high", 0, 255, key="det_val_high")
        st.number_input("Min contour area (px²)", 1, 5000, key="det_min_area")
        st.number_input("Max contour area (px²)", 10, 50000, key="det_max_area")
        st.slider("Min circularity", 0.0, 1.0, key="det_min_circ",
                  help="4·π·A/P² — 1.0 is a perfect circle.")
        st.number_input("Max ball radius (px)", 3, 120, key="det_max_radius")
        st.slider("Min motion overlap", 0.0, 0.9, key="det_min_motion",
                  help="Hard gate: a candidate must overlap the motion mask "
                       "by at least this fraction, which rejects static "
                       "look-alikes (table lines, banner text). Requires "
                       "motion evidence to be on. Raise it if the tracker "
                       "locks onto players' shoes or static objects.")
    with st.expander("Advanced · motion & tracking"):
        st.checkbox("Use motion evidence", key="det_use_motion")
        st.selectbox("Motion source", ["diff", "mog2"], key="det_motion_source",
                     format_func=lambda v: "Frame differencing" if v == "diff"
                     else "MOG2 background subtraction")
        st.slider("Frame-diff threshold", 5, 80, key="det_diff_threshold")
        st.number_input("Missed-frame limit (reset after)", 1, 60,
                        key="trk_max_missed",
                        help="Coast on Kalman prediction this long, then reset.")
        st.number_input("Gating radius (px)", 20, 600, key="trk_max_jump",
                        help="Candidates farther than this from the "
                             "prediction are ignored.")
        st.number_input("Measurement noise (px)", 0.5, 20.0,
                        key="trk_meas_noise")
        st.number_input("Process noise", 100, 20000,
                        key="trk_process_noise")
    with st.expander("Advanced · bounce heuristic & speed"):
        st.number_input("Bounce cooldown (frames)", 1, 120, key="bnc_cooldown")
        st.number_input("Min reversal strength (px/frame)", 0.5, 30.0,
                        key="bnc_min_strength")
        st.number_input("Table margin (m)", 0.05, 2.0, key="bnc_table_margin")
        st.slider("Uncertain-bounce threshold", 0.0, 1.0,
                  key="bnc_uncertain",
                  help="Heuristic confidence below this marks an event as "
                       "'uncertain'.")
        st.number_input("Speed max gap (frames)", 1, 30, key="spd_max_gap",
                        help="No speed is computed across tracking gaps "
                             "longer than this.")
        st.number_input("Speed smoothing window", 1, 21, key="spd_smooth")


# ---------------------------------------------------------------- stage 1
stage_header(1, "Video", "Upload a fixed-camera recording of one ball — "
                         "MP4/MOV/M4V/MKV/AVI/WEBM, up to 400 MB. The file "
                         "never leaves this machine.")
uploaded = st.file_uploader("Video file", type=["mp4", "mov", "m4v", "mkv",
                                                "avi", "webm"],
                            label_visibility="collapsed")

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
MAX_BYTES = 400 * 1024 * 1024

if uploaded is not None:
    same = (st.session_state.get("upload_name") == uploaded.name
            and st.session_state.get("upload_size") == uploaded.size)
    if not same:
        # Validate the new file completely before discarding previous results.
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            st.error(f"Unsupported file type '{suffix}'.")
            st.stop()
        if uploaded.size > MAX_BYTES:
            st.error("File larger than 400 MB — please upload a shorter clip.")
            st.stop()
        tmp = new_artifact_dir("upload")
        dest = tmp / f"input{suffix}"
        try:
            with open(dest, "wb") as fh:
                while True:
                    chunk = uploaded.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            info = probe_video(str(dest))
        except Exception as exc:  # unreadable / unsupported codec
            shutil.rmtree(tmp, ignore_errors=True)
            st.error(f"Could not read this video: {exc}")
            st.stop()
        for key in ("video_path", "video_info", "rep_frame", "calibration",
                    "preview", "analysis", "calib_skipped", "resolved_color",
                    "cx0", "cy0", "cx1", "cy1", "cx2", "cy2", "cx3", "cy3"):
            st.session_state.pop(key, None)
        old = st.session_state.pop("artifacts_dir", None)
        if old and Path(old).exists():
            shutil.rmtree(old, ignore_errors=True)
        st.session_state.update(
            upload_name=uploaded.name, upload_size=uploaded.size,
            video_path=str(dest), video_info=info,
            rep_frame=grab_frame(str(dest),
                                 max(info.frame_count // 2, 0) if info.frame_count else 0))
        st.session_state["artifacts_dir"] = str(tmp)

if "video_path" in st.session_state:
    info = st.session_state["video_info"]
    rep: Optional[np.ndarray] = st.session_state.get("rep_frame")
    left, right = st.columns([1.7, 1])
    with left:
        if rep is not None:
            st.image(cv2.cvtColor(rep, cv2.COLOR_BGR2RGB),
                     caption=(f"Representative frame of "
                              f"“{st.session_state.get('upload_name', 'video')}” "
                              "(middle of the file)"),
                     use_container_width=True)
        else:
            st.error("Could not decode a frame from this video.")
    with right:
        m1, m2 = st.columns(2)
        with m1:
            metric_card("Resolution", f"{info.width}×{info.height}")
        with m2:
            metric_card("Frame rate", f"{info.fps:g}",
                        "default assumption" if info.fps_is_default
                        else "from metadata")
        m3, m4 = st.columns(2)
        with m3:
            metric_card("Frames", f"{info.frame_count:,}" if info.frame_count
                        else "unknown")
        with m4:
            metric_card("Duration",
                        f"{info.duration_s:.1f} s" if info.duration_is_known
                        else "unknown")
        metric_card("Timestamp source", info.timestamp_source,
                    "container timestamps" if info.timestamp_source ==
                    "container-msec" else "frame index ÷ fps")
        for w in info.warnings:
            st.warning(w)
else:
    st.markdown("""
<div class="panel">
  <b>How it works</b>
  <ol>
    <li><b>Video</b> — upload a fixed-camera recording (processed locally).</li>
    <li><b>Calibration</b> — enter the four table corners, or skip for
        pixel-space results.</li>
    <li><b>Preview</b> — check detection on the first seconds and tune
        parameters.</li>
    <li><b>Analysis</b> — annotated video, estimated bounces, speeds and
        downloadable results.</li>
  </ol>
  <span class="small-note">Best results: fixed camera · one white or orange
  ball · even lighting · 60 fps or higher · all four table corners
  visible.</span>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- stage 2
if "video_path" in st.session_state:
    st.divider()
    stage_header(2, "Calibration", "Mark the four table corners in the "
                                    "documented order — or skip and get "
                                    "pixel-space results.")
    rep = st.session_state.get("rep_frame")
    info = st.session_state["video_info"]
    cal = st.session_state.get("calibration")
    if cal is not None:
        st.success("Calibration active. The white outline below is the "
                   "projected table region (for reference only).")

    left, right = st.columns([3, 2])
    with right:
        st.markdown("**Corner input order (clockwise as seen in the image):**")
        for i, name in enumerate(CORNER_ORDER):
            st.markdown(f"**{i+1} · {name}** — {CORNER_GUIDE[name]}")
        st.markdown(
            "Corner 1→2 is mapped to the **table length** (x axis) and 2→3 "
            "to the **width** (y axis). If the top edge in your framing is a "
            "short edge, swap the length/width values accordingly.")
        L = st.number_input("Table length (m)", 0.5, 10.0, 2.74, 0.01,
                            key="table_length_m")
        W = st.number_input("Table width (m)", 0.5, 10.0, 1.525, 0.01,
                            key="table_width_m")

        guess = dict(c0=(0.25, 0.35), c1=(0.75, 0.35), c2=(0.75, 0.75),
                     c3=(0.25, 0.75))
        pts = []
        cols = st.columns([2, 1, 1])
        cols[1].markdown("`x (px)`"); cols[2].markdown("`y (px)`")
        for i, name in enumerate(CORNER_ORDER):
            gx = int(info.width * guess[f"c{i}"][0])
            gy = int(info.height * guess[f"c{i}"][1])
            row = st.columns([2, 1, 1])
            row[0].markdown(f"**{i+1} · {name}**")
            x = row[1].number_input(f"x{i}", 0.0, float(info.width), float(gx),
                                    1.0, key=f"cx{i}", label_visibility="collapsed")
            y = row[2].number_input(f"y{i}", 0.0, float(info.height), float(gy),
                                    1.0, key=f"cy{i}", label_visibility="collapsed")
            pts.append((x, y))
        a1, a2 = st.columns(2)
        apply = a1.button("Apply calibration", type="primary",
                          use_container_width=True)
        skip = a2.button("Skip (pixel-space only)",
                         use_container_width=True)
        if skip:
            st.session_state["calibration"] = None
            st.session_state["calib_skipped"] = True
        if apply:
            try:
                st.session_state["calibration"] = TableCalibration(
                    [list(p) for p in pts], float(L), float(W))
                st.session_state["calib_skipped"] = False
                st.rerun()
            except CalibrationError as exc:
                st.error(f"Invalid calibration: {exc}")

    with left:
        if rep is not None:
            disp = rep.copy()
            fh_, fw_ = disp.shape[:2]
            fs = max(0.38, fw_ / 2600.0)          # label font scale
            step = next(s for s in (25, 50, 100, 200, 400) if s >= fw_ / 24)
            for gx in range(step, fw_, step):
                cv2.line(disp, (gx, 0), (gx, fh_), (205, 205, 205), 1)
                cv2.putText(disp, str(gx), (gx + 2, int(18 * fs / 0.38)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (110, 110, 110), 1)
            for gy in range(step, fh_, step):
                cv2.line(disp, (0, gy), (fw_, gy), (205, 205, 205), 1)
                cv2.putText(disp, str(gy), (2, gy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (110, 110, 110), 1)
            cal_now = st.session_state.get("calibration")
            if cal_now is not None:
                draw_table_overlay(disp, cal_now)
                corners = np.array(cal_now.image_points, np.int32)
            else:
                corners = np.array(pts, np.int32)
            marker = max(16, int(fw_ / 80))
            for i, (cx, cy) in enumerate(corners):
                cv2.drawMarker(disp, (int(cx), int(cy)), (0, 102, 255),
                               cv2.MARKER_DIAMOND, marker,
                               max(2, int(fs * 5)))
                cv2.putText(disp, str(i + 1),
                            (int(cx) + marker // 2 + 2, int(cy) - marker // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 1.8, (0, 102, 255),
                            max(2, int(fs * 4)))
            st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB),
                     caption=(f"Frame with {step}-px grid — read pixel "
                              "coordinates and enter them on the right."),
                     use_container_width=True)
        if st.session_state.get("calib_skipped") and cal is None:
            st.info("Calibration skipped: speeds will be reported in px/s and "
                    "bounce table-proximity cannot be verified.")

# ---------------------------------------------------------------- stage 3
if "video_path" in st.session_state:
    st.divider()
    stage_header(3, "Preview", "Quick check on a short segment before the "
                               "full run.")
    info = st.session_state["video_info"]
    n_prev = min(int(4 * info.fps), 240)
    if info.frame_count:
        n_prev = min(n_prev, info.frame_count)
    run = st.button(f"Run preview (first {n_prev} frames)",
                    type="primary")
    if run:
        params = build_params()
        bar = st.progress(0.0, "Detecting…")
        try:
            det_params = params["detection"]
            if st.session_state["ball_color"] == "auto":
                st.info("Ball colour is **auto** — trying the orange and "
                        "white presets on the preview segment…")
                winner, attempts = auto_pick_color(
                    st.session_state["video_path"], info,
                    st.session_state.get("calibration"), params["tracking"],
                    params["bounce"], params["speed"], n_frames=n_prev)
                st.session_state["resolved_color"] = winner
                det_params = preset_params(winner)
                so = attempts["orange"].stats
                sw = attempts["white"].stats
                st.success(
                    f"Auto picked **{winner}** · orange: "
                    f"{so['measured_coverage']*100:.0f}% coverage, "
                    f"{so['median_pixel_speed_pxps'] or 0:.0f} px/s median · "
                    f"white: {sw['measured_coverage']*100:.0f}% coverage, "
                    f"{sw['median_pixel_speed_pxps'] or 0:.0f} px/s median.")
                preview = attempts[winner]
            else:
                analyzer = VideoAnalyzer(
                    st.session_state["video_path"],
                    calibration=st.session_state.get("calibration"),
                    video_info=info, detection=det_params,
                    tracking=params["tracking"], bounce=params["bounce"],
                    speed=params["speed"])
                preview = analyzer.analyze(
                    max_frames=n_prev,
                    progress_cb=lambda n: bar.progress(
                        min(n / n_prev, 1.0), f"Frame {n}/{n_prev}"))
            st.session_state["preview"] = preview
        except Exception as exc:
            st.error(f"Preview failed: {exc}")
        bar.empty()
    prev_res = st.session_state.get("preview")
    if prev_res is not None:
        frames = render_preview_frames(st.session_state["video_path"], prev_res)
        for i in range(0, len(frames), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(frames):
                    f_idx, img = frames[i + j]
                    col.image(img, caption=f"frame {f_idx}",
                              use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Measured coverage",
                        f"{prev_res.measured_coverage()*100:.0f}%",
                        "detected / processed frames in preview")
        with c2:
            metric_card("Estimated bounces",
                        f"{prev_res.stats['estimated_bounces_total']}",
                        f"{prev_res.stats['estimated_bounces_certain']} certain · "
                        f"{prev_res.stats['estimated_bounces_uncertain']} uncertain "
                        "(heuristic)")
        with c3:
            metric_card("Median speed",
                        (f"{fmt(prev_res.stats['median_projected_speed_mps'])} m/s"
                         if prev_res.calibration
                         else f"{fmt(prev_res.stats['median_pixel_speed_pxps'], '.0f')} px/s"),
                        "approx. table-plane projected" if prev_res.calibration
                        else "pixel speed (uncalibrated)")
        rep = st.session_state.get("rep_frame")
        if rep is not None:
            m = colour_mask_preview(rep, build_params()["detection"])
            st.image(cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), caption=(
                "Colour mask on the representative frame — use it to tune the "
                "HSV sliders in the sidebar."), use_container_width=True)
        if prev_res.measured_coverage() < 0.3:
            st.warning("Low measured coverage. Try adjusting the HSV range, "
                       "loosening area/circularity filters, or the motion "
                       "source in the sidebar.")

# ------------------------------------------------------------------ charts
def trajectory_fig(result: AnalysisResult) -> go.Figure:
    meas = [(r.x, r.y) for r in result.records if r.status == "detected"]
    pred = [(r.x, r.y) for r in result.records if r.status == "predicted"]
    fig = go.Figure()
    if meas:
        fig.add_trace(go.Scatter(x=[p[0] for p in meas], y=[p[1] for p in meas],
                                 mode="markers", name="measured",
                                 marker=dict(size=4, color="#2CA02C")))
    if pred:
        fig.add_trace(go.Scatter(x=[p[0] for p in pred], y=[p[1] for p in pred],
                                 mode="markers", name="predicted (coasted)",
                                 marker=dict(size=4, color=ACCENT,
                                             symbol="circle-open")))
    if result.bounces:
        fig.add_trace(go.Scatter(x=[b.img_x for b in result.bounces],
                                 y=[b.img_y for b in result.bounces],
                                 mode="markers", name="estimated bounce",
                                 marker=dict(size=11, color="#D62728",
                                             symbol="x")))
    fig.update_layout(title="Ball trajectory (image coordinates)",
                      xaxis_title="x (px)", yaxis_title="y (px)",
                      yaxis=dict(autorange="reversed"),
                      template="plotly_white", height=430,
                      margin=dict(l=10, r=10, t=45, b=10))
    return fig


def table_fig(result: AnalysisResult) -> go.Figure:
    cal = result.calibration
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=cal.length_m, y1=cal.width_m,
                  line=dict(color="#999", width=1.5))
    fig.add_shape(type="line", x0=cal.length_m / 2, y0=0,
                  x1=cal.length_m / 2, y1=cal.width_m,
                  line=dict(color="#bbb", width=1, dash="dot"))
    pts = [(r.meas_x, r.meas_y) for r in result.records
           if r.status == "detected" and math.isfinite(r.meas_x)]
    if pts:
        xs, ys = cal.project_many(np.array(pts))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="measured",
                                 marker=dict(size=3.5, color="#2CA02C")))
    if result.bounces and math.isfinite(result.bounces[0].table_x):
        fig.add_trace(go.Scatter(x=[b.table_x for b in result.bounces],
                                 y=[b.table_y for b in result.bounces],
                                 mode="markers", name="estimated bounce",
                                 marker=dict(size=10, color="#D62728",
                                             symbol="x")))
    fig.update_layout(title="Projected onto the table plane (approximation — "
                            "airborne points are off-plane)",
                      xaxis_title="table x (m)", yaxis_title="table y (m)",
                      template="plotly_white", height=430,
                      margin=dict(l=10, r=10, t=45, b=10))
    return fig


def speed_fig(result: AnalysisResult) -> go.Figure:
    speeds, speeds_px, _ = compute_speeds(result.records, result.calibration,
                                          result.speed_params)
    ts = [r.t for r in result.records]
    unit = "m/s (approx. table-plane projected)" if result.calibration else "px/s"
    vals = speeds if result.calibration else speeds_px
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines+markers", name="speed",
                             marker=dict(size=3.5), line=dict(color=ACCENT,
                                                              width=1.5)))
    fin = vals[np.isfinite(vals)]
    if fin.size:
        fig.add_hline(y=float(np.median(fin)), line_dash="dot",
                      line_color="#888",
                      annotation_text=f"median {np.median(fin):.1f}")
    for ev in result.bounces:
        fig.add_vline(x=ev.t, line_color="#D62728" if ev.certain else "#F2A6A6",
                      line_width=1,
                      annotation_text="bounce", annotation_font_size=9)
    fig.update_layout(title=f"Speed vs time — {unit}",
                      xaxis_title="time (s)", yaxis_title=unit,
                      template="plotly_white", height=380,
                      margin=dict(l=10, r=10, t=45, b=10))
    if result.calibration is not None:
        fig.update_layout(
            yaxis2=dict(title="km/h", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(orientation="h", y=1.12))
    return fig


# ---------------------------------------------------------------- stage 4
if "video_path" in st.session_state:
    st.divider()
    stage_header(4, "Analysis", "Full pass with progress, annotated video, "
                                "plots and downloads.")
    cal = st.session_state.get("calibration")
    if cal is None and not st.session_state.get("calib_skipped"):
        st.warning("No calibration applied — metric speeds are disabled and "
                   "results will be in pixel space. You can still proceed.")
    run = st.button("Analyze full video", type="primary")
    if run:
        params = build_params()
        info = st.session_state["video_info"]
        art = new_artifact_dir("result")
        old = st.session_state.get("result_artifacts")
        bar = st.progress(0.0, "Pass 1/2 — detection & tracking…")
        try:
            det_params = params["detection"]
            if st.session_state["ball_color"] == "auto":
                winner = st.session_state.get("resolved_color")
                if winner is None:
                    winner, _ = auto_pick_color(
                        st.session_state["video_path"], info, cal,
                        params["tracking"], params["bounce"],
                        params["speed"])
                    st.session_state["resolved_color"] = winner
                det_params = preset_params(winner)
                st.info(f"Ball colour **auto** → using the **{winner}** "
                        "preset (resolved from the preview segment).")
            analyzer = VideoAnalyzer(st.session_state["video_path"],
                                     calibration=cal, video_info=info,
                                     detection=det_params,
                                     tracking=params["tracking"],
                                     bounce=params["bounce"],
                                     speed=params["speed"])
            total = info.frame_count or None
            def pcb1(n):
                frac = min(n / total, 1.0) if total else min(n / (n + 500), 1.0)
                bar.progress(0.6 * frac, f"Pass 1/2 — frame {n}")
            result = analyzer.analyze(progress_cb=pcb1)
            out_mp4 = str(art / "annotated.mp4")
            def pcb2(n):
                frac = min(n / total, 1.0) if total else min(n / (n + 500), 1.0)
                bar.progress(0.6 + 0.4 * frac, f"Pass 2/2 — annotating frame {n}")
            annotate_video(st.session_state["video_path"], out_mp4, result,
                           progress_cb=pcb2)
            st.session_state["analysis"] = dict(
                result=result, video_path=out_mp4,
                artifacts=str(art), fourcc=None,
                df_frames=per_frame_dataframe(result),
                df_bounces=bounces_dataframe(result),
                summary=summary_json(result, writer_fourcc="see file metadata"))
            st.session_state["result_artifacts"] = str(art)
            if old and old != str(art):
                shutil.rmtree(old, ignore_errors=True)
            bar.progress(1.0, "Done.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
        finally:
            bar.empty()

    res = st.session_state.get("analysis")
    if res is not None:
        result: AnalysisResult = res["result"]
        if result.processed_frames == 0:
            st.error("No frames could be decoded from this video — nothing "
                     "to analyse. Try a different file or codec.")
            st.stop()
        if result.stats["measured_frames"] == 0:
            st.error("The ball was never detected. Re-run the preview stage "
                     "and adjust the detection parameters in the sidebar.")
            st.stop()
        s = result.stats
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Duration", f"{s['duration_s']:.1f} s",
                        f"{s['processed_frames']:,} frames")
        with c2:
            metric_card("Measured coverage", f"{s['measured_coverage']*100:.0f}%",
                        f"{s['measured_frames']:,} measured · "
                        f"{s['predicted_frames']:,} predicted (not used as evidence)")
        with c3:
            metric_card("Estimated bounces", f"{s['estimated_bounces_total']}",
                        f"{s['estimated_bounces_certain']} certain · "
                        f"{s['estimated_bounces_uncertain']} uncertain "
                        "(heuristic score)")
        if result.calibration is not None:
            with c4:
                metric_card("Median projected speed",
                            f"{fmt(s['median_projected_speed_mps'])} m/s",
                            "approx. table-plane projected · not 3D speed")
            with c5:
                metric_card("Max projected speed",
                            f"{fmt(s['max_projected_speed_mps'])} m/s "
                            f"({fmt(s['max_projected_speed_mps'] * 3.6, '.0f')} km/h)",
                            "after outlier filtering")
        else:
            with c4:
                metric_card("Median pixel speed",
                            f"{fmt(s['median_pixel_speed_pxps'], '.0f')} px/s",
                            "uncalibrated — metric speed disabled")
            with c5:
                metric_card("Max pixel speed",
                            f"{fmt(s['max_pixel_speed_pxps'], '.0f')} px/s",
                            "after outlier filtering")
        st.caption("Speed = **approximate table-plane projected speed**: the "
                   "homography is exact only on the table surface, and an "
                   "airborne ball lies off that plane.")

        tabs = st.tabs(["Annotated video", "Trajectory", "Speed",
                        "Bounce events", "Downloads & summary"])
        with tabs[0]:
            st.video(res["video_path"])
            st.caption("Green = measured · orange ring = Kalman-predicted "
                       "(coasted) · red marker = estimated bounce. If your "
                       "browser cannot play this MP4 codec, use the download "
                       "button instead.")
        with tabs[1]:
            st.plotly_chart(trajectory_fig(result), use_container_width=True)
            if result.calibration is not None:
                st.plotly_chart(table_fig(result), use_container_width=True)
        with tabs[2]:
            st.plotly_chart(speed_fig(result), use_container_width=True)
        with tabs[3]:
            if res["df_bounces"].empty:
                st.info("No estimated bounce events passed the heuristic "
                        "filters. Try lowering the reversal-strength threshold "
                        "or widening the table margin in the sidebar — or the "
                        "ball may genuinely not have bounced in this clip.")
            else:
                st.dataframe(res["df_bounces"], use_container_width=True,
                             hide_index=True)
                st.caption("confidence_heuristic is a transparent heuristic "
                           "score (table proximity, reversal strength, "
                           "measured coverage) — not a calibrated probability.")
        with tabs[4]:
            b1, b2 = st.columns(2)
            with b1:
                st.download_button("Download per-frame CSV",
                                   res["df_frames"].to_csv(index=False).encode(),
                                   "per_frame.csv", "text/csv",
                                   use_container_width=True)
                st.download_button("Download bounce events CSV",
                                   res["df_bounces"].to_csv(index=False).encode(),
                                   "bounce_events.csv", "text/csv",
                                   use_container_width=True)
            with b2:
                st.download_button("Download JSON summary",
                                   res["summary"], "summary.json",
                                   "application/json", use_container_width=True)
                with open(res["video_path"], "rb") as fh:
                    st.download_button("Download annotated video", fh.read(),
                                       "annotated.mp4", "video/mp4",
                                       use_container_width=True)
            st.markdown("**Measurement limitations**")
            for lim in LIMITATIONS:
                st.markdown(f"- {lim}")
            with st.expander("JSON summary"):
                st.code(res["summary"], language="json")
