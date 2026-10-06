# Vision-Based Table Tennis Ball Tracking, Bounce Detection and Speed Estimation

A local Streamlit application for a college Image & Video Processing project.
It tracks a single table-tennis ball in a **fixed-camera recorded video** using
classical image processing (no training, no pretrained models, no external
datasets), estimates **heuristic bounce events**, and reports an
**approximate table-plane projected speed** — honestly labelled as such, not a
true 3D ball speed.

> **What this is not:** it is not a professional sports measurement system, it
> does no 3D reconstruction, and it is not an automatic fault/service umpire.
> All speeds are approximations; all bounce events are heuristic estimates.

---

## 1. Setup and launch

Requires Python 3.10+ (tested on 3.13 with OpenCV 5.0).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # only needed to run the tests

streamlit run app.py
```

Then open the printed local URL (default <http://localhost:8501>). Everything
runs locally: uploads are written to a temporary directory on your machine and
nothing is sent anywhere. Uploads up to **400 MB** are accepted
(`.streamlit/config.toml` → `server.maxUploadSize`; restart the app after
changing it).

No real recording handy? Generate a clearly-synthetic demo clip (dark room,
table, one orange ball bouncing at 2 s and 5 s):

```bash
python scripts/make_demo_video.py      # writes demo_clip.mp4
```

Suggested calibration corners for this clip are documented in the script's
docstring.

**Video codec note:** the annotated video is written with OpenCV's bundled
encoder (`avc1`/H.264 when available, otherwise `mp4v`). Both play in most
browsers or via the download button. FFmpeg is **not** required and not
auto-invoked; if your OpenCV build could not produce a browser-compatible
codec, download the file and convert it yourself — the app does not crash,
it just tells you the codec it used.

## 2. Recommended recording conditions

- **Fixed camera** on a tripod; the app assumes a static scene.
- Good, even lighting; avoid backlight and flickering (cheap LED/fluorescent
  lights at slow shutter cause flicker).
- One ball in view; white or orange. The table surface should be a distinctly
  different colour from the ball (a blue or green table helps a lot).
- Frame rate: 60 fps or higher gives much better speed and bounce estimates;
  30 fps works but with coarser resolution.
- Frame the whole table so all four corners are visible and mostly unoccluded.
- Keep the ball reasonably large in frame (a ball radius of 3–15 px is the
  sweet spot for the default filters).

## 3. Using the app

The dashboard has four numbered stages:

1. **Video** — upload an MP4/MOV/MKV/AVI/WEBM file (decoding depends on your
   OpenCV build). The app shows metadata (resolution, FPS, header frame
   count, duration, timestamp source) and a representative mid-file frame.
   Warnings appear when FPS metadata is missing or container timestamps look
   unreliable.
2. **Calibration** — read the pixel coordinates of the four table corners off
   the gridded frame and enter them in the documented order (below), then
   **Apply calibration**. Or press **Skip (pixel-space only)**: results will
   then be reported in px/s and metric speeds stay disabled.
3. **Preview** — press the button to run detection + tracking on the first
   ~4 seconds. Inspect the annotated frames, coverage and bounce count, and
   tune the HSV sliders in the sidebar (the colour-mask preview shows exactly
   what the segmentation sees) before committing to the full run.
4. **Analysis** — press **Analyze full video**. The pipeline streams the video
   in two passes (detect/track, then annotate) with progress bars, and shows
   the annotated video, trajectory plots, speed plot, bounce table and
   download buttons (per-frame CSV, bounce CSV, JSON summary, annotated MP4).

Nothing heavy runs while you move sliders — processing happens only behind
the explicit Preview / Analyze buttons.

### Tuning tips for real footage

- **Leave Ball colour on `auto`** unless you have a reason not to: the
  preview tries both colour presets on the segment and keeps the one that
  tracks a faster, more consistent object (a ball, not a shoe or the floor).
  Pick a colour explicitly only to hand-tune the sliders.
- **Marker sits on a shoe, the floor, or a table line?** Switch **Ball
  colour** to the actual ball colour first (the presets differ a lot), then
  raise **Min motion overlap** in the sidebar (0.2–0.3 rejects anything
  moving slower than a ball).
- **Very low measured coverage in the preview?** The ball is likely
  motion-blurred: lower *Value low* (≈175 for blurred 120 fps white balls),
  raise *Max contour area*, and lower *Min circularity* a little.
- **Locks onto a static bright spot?** That means the motion gate is off or
  too low — check *Use motion evidence* is on and raise *Min motion
  overlap*.
- Judge every change by the preview stage's annotated frames, coverage card
  and colour-mask image before running the full analysis.

### Calibration point order

Enter the four corners **clockwise as seen in the image**:

```
  1 Top-left  ──►  2 Top-right      (edge 1→2 becomes the table LENGTH, x axis)
      ▲                │
      │                ▼
  4 Bottom-left ◄─  3 Bottom-right  (edge 2→3 becomes the table WIDTH, y axis)
```

- **1 · Top-left** — upper-most left corner of the table surface in the image
- **2 · Top-right** — upper-most right corner
- **3 · Bottom-right** — lower-most right corner
- **4 · Bottom-left** — lower-most left corner

Defaults are 2.74 m × 1.525 m (standard table); both are editable. If the
long edge of your table runs vertically in the frame, swap the length/width
values. The app rejects degenerate input (duplicate/collinear corners).

## 4. Image & video processing techniques (what the pipeline does)

Per frame, in `src/`:

1. **Colour segmentation** (`detection.py`) — the frame is converted to HSV
   and thresholded with an adjustable range (presets for white and orange
   balls; every bound is tunable in the sidebar).
2. **Motion evidence** — frame differencing (absolute difference of
   consecutive blurred grey frames, thresholded and dilated) or MOG2
   background subtraction supplies an independent motion mask.
3. **Morphological cleanup** — open then close with small elliptical kernels
   removes speckle and fills the ball blob.
4. **Contour filtering** — connected components are filtered by area range,
   circularity 4πA/P², and an equivalent-radius (from area) plausible range.
5. **Candidate scoring** — every surviving candidate is scored with
   normalised appearance terms (circularity, solidity), motion overlap, and a
   Gaussian distance term toward the tracker's prediction. A hard
   **minimum-motion-overlap gate** (default 0.15) additionally rejects
   anything that is not actually moving — table lines, banner text, shoes at
   rest. The **largest contour is never blindly chosen**; multiple candidates
   are ranked and the tracker gates them itself.
6. **Tracking** (`tracking.py`) — a constant-velocity Kalman filter over
   state [x, y, vx, vy] in pixels and px/second. The transition matrix is
   rebuilt from the real per-frame dt (from container timestamps or
   frame-index/FPS), so prediction is frame-rate aware. Candidates are gated
   by a maximum jump radius around the prediction. On loss the filter coasts
   ("predicted") for a configurable missed-frame limit, then resets;
   reacquisition requires a configurable number of consistent consecutive
   detections near the same spot (a single stray detection cannot re-lock).
7. **Calibration** (`calibration.py`) — a homography from the four image
   corners to a rectangle in metres, computed with `cv2.findHomography`.
8. **Bounce heuristic** (`bounce.py`) — see §6.
9. **Exports** (`export.py`) — annotated video (second streaming pass),
   plots (Plotly in the UI), CSVs and a JSON summary.

Every recorded frame is labelled **detected**, **predicted** (Kalman coast)
or **lost**. Predicted points keep the visual track continuous but are
**never used as evidence** for speed or bounce confirmation.

## 5. Speed estimation — and its honest limits

- Speeds are computed **only from timestamped, measured positions** of
  consecutive detected frames, never from Kalman predictions.
- Gaps longer than a configurable number of frames produce **no** speed
  value.
- A leave-one-out **Hampel outlier filter** (robust median/MAD) removes wild
  speeds before a modest centred rolling-median smoothing. Smoothing never
  invents values for missing frames.
- With calibration, the two endpoints of each measured displacement are
  projected through the homography and the resulting speed is displayed as
  **"approximate table-plane projected speed"** in m/s and km/h.
- Without valid calibration, speeds are shown in **px/s** and metric speeds
  are disabled.

**The key limitation:** the homography is exact **only on the table plane**.
A ball in flight lies above that plane, so its projected position is the
perspective "shadow" of a 3D point — its projected speed is distorted by the
ball's height (a high fast ball can appear slower; a low slow ball can appear
faster, depending on camera geometry). This is why the UI always says
"approximate table-plane projected speed" and never "true 3D speed".

Timestamps come from the container (POS_MSEC) when the probe finds them
monotonic and consistent with the header FPS; otherwise frame index divided
by source FPS. The *processing* frame rate never enters any calculation.

## 6. Bounce detection — a transparent heuristic

For each measured frame the heuristic (`src/bounce.py`):

1. Smooths short windows of the measured y trajectory (image y points down).
2. Estimates the vertical slope before and after the candidate frame over
   configurable windows (px/frame, so thresholds are frame-rate independent).
3. Flags a **downward→upward reversal** whose combined strength exceeds a
   threshold, located (when calibrated) near the table quad plus a margin;
   reversals far from the table plane are rejected outright.
4. Requires a minimum number of measured detections around the event and
   rejects candidates adjacent to tracking gaps longer than the limit.
5. Applies a configurable **cooldown** so one physical bounce is not counted
   several times (the strongest candidate in a cluster wins).
6. Scores each event with a transparent blend of table proximity, reversal
   strength and measured coverage. Below the "uncertain" threshold the event
   is labelled **uncertain** in the UI and exports.

**Limits:** a vertical reversal alone does not prove a bounce. Racket
contacts, net-cord deflections, occlusion drop-outs and unusual camera
angles can each create false positives or misses. Two bounces only a few
frames apart cannot be resolved (the velocity windows are wider than the
separation). The confidence score is a **heuristic, not a calibrated
probability**. Side-view and high-behind-camera placements work best;
near-table-level side views compress the vertical motion and weaken the
signal.

## 7. Outputs

- **Annotated MP4** — measured (green) vs predicted (orange ring) positions,
  short trail, tracking status/time overlay, per-frame projected speed text,
  table outline, and estimated bounce markers with their heuristic
  confidence.
- **Trajectory plot** (image coordinates, measured/predicted/bounces) and, if
  calibrated, a **table-plane plot** with the net line drawn.
- **Speed-vs-time plot** with median line and bounce markers.
- **Per-frame CSV** — timestamp (and its source), status, measured/predicted
  coordinates, detection score, filter velocity, projected table coordinates,
  speed in m/s & km/h & px/s, outlier flag.
- **Bounce-events CSV** — frame, timestamp, image coords, projected table
  coords, vy before/after, heuristic confidence, certain/uncertain.
- **JSON summary** — video metadata and warnings, settings, calibration,
  aggregate stats (duration, measured coverage, bounce counts, median & max
  projected speed after outlier filtering) and the full list of measurement
  limitations.

## 8. Evaluating against manual labels

The synthetic tests prove the *plumbing*, not real-world accuracy. To
evaluate properly:

1. **Bounce frames:** step through your video in any player, note the frame
   index of every ground truth bounce (the annotated video's frame overlay
   helps). Export the bounce CSV and count, for some tolerance window of ±k
   frames (try k = 3 and 5): true positives (estimated event within k of a
   labelled bounce), false positives, misses. Report precision and recall —
   not accuracy percentages alone.
2. **Ball positions:** click the ball centre in a sample of frames (e.g. in
   an image editor that shows pixel coordinates), compare with
   `measured_x_px`/`measured_y_px` in the per-frame CSV, and report the mean
   absolute error in px.
3. **Speed:** for one rally, divide known displacements (e.g. table length
   crossings on the surface) by the time between the corresponding frames in
   the CSV, and compare with the reported projected speed for those frames.
   Expect differences whenever the ball is airborne — that is the plane
   projection error described above, not necessarily a bug.
4. Vary camera placement (side view vs high view) and note how the bounce
   heuristic and projected speeds degrade — this is expected behaviour.

## 9. Verification performed (and not performed)

Performed in this environment:

- `python -m pytest tests/ -q` — **28 tests pass**, all on synthetic data:
  - Homography exactness on known corners; degenerate/duplicate/collinear
    calibration rejection (`tests/test_calibration.py`).
  - Speed from known positions + timestamps; pixel-only mode; outlier spike
    rejection; no speed across long gaps; predicted points never used as
    speed evidence (`tests/test_speed.py`).
  - Tracker follow/coast/reset after the missed-frame limit; reacquisition
    confirmation; prediction gating; Kalman dt-awareness
    (`tests/test_tracking.py`).
  - Bounce detection near the true event; cooldown/duplicate suppression and
    threshold behaviour; gap rejection; no-evidence-no-events; table-proximity
    rejection and confidence behaviour (`tests/test_bounce.py`).
  - End-to-end pipeline on a generated video (probe → track → bounce →
    annotated MP4 round-trip) (`tests/test_pipeline.py`).
- Application startup verified with a headless Streamlit server (health check
  + HTTP 200) and the script body executed via Streamlit's AppTest harness.
- The demo clip runs through the full pipeline: 100 % measured coverage and
  both synthetic bounces detected at their true frames (60 and 150).

**Not performed:** validation on real recorded table-tennis videos (none was
available), so real-world detection robustness, bounce precision/recall and
speed accuracy are **unknown and deliberately not quantified here**. All
numbers above come from generated data and say nothing about real-world
accuracy. No ffmpeg was installed; the annotated-video codec path relies on
OpenCV's bundled encoder (verified to write `avc1` MP4 in this environment).

## 10. Project layout

```
app.py                  Streamlit dashboard (stages 1–4)
src/detection.py        HSV + motion + morphology + contour scoring
src/tracking.py         Kalman (constant velocity), gating, reacquisition
src/calibration.py      Corner validation, homography, projection
src/bounce.py           Transparent bounce heuristic
src/analysis.py         Video probe, streaming pass 1, speeds, summary
src/export.py           Annotated video (pass 2), CSVs, JSON summary
scripts/make_demo_video.py  Synthetic demo clip generator
tests/                  Synthetic-data pytest suite
requirements.txt        Runtime dependencies
requirements-dev.txt    Test dependencies
```
