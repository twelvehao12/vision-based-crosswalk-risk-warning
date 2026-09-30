"""Calibrate a ROI file for an arbitrary camera or video source.

`make_webcam_roi.py` can only move an existing calibration onto a new frame.
This tool starts from the scene itself: it shows a frame from the camera or
video, you draw each region the pipeline needs, and it writes a ROI file that
`main.py` and `run_webcam.py` accept as-is.

The frame is frozen while you draw, because a live camera would shift the
scene under every click. Press `f` (or space) to peek at the live feed.

Regions are drawn in the order the pipeline needs them:

    1 crosswalk_roi                   the crossing surface
    2 vehicle_approach_zone           the lane vehicles come from
    3 pedestrian_waiting_zone         where pedestrians wait
    4 secondary_crosswalk_roi         optional second crossing
    5 secondary_vehicle_approach_zone optional approach for the second one

Left click adds a point, right click (or `u`) undoes one, `c` clears the active
region, `n` moves to the next region and `s` saves. The two secondary regions
are only written together, because the pipeline rejects one without the other.

Usage:
    python tools/calibrate_roi.py --source 0
    python tools/calibrate_roi.py --source 0 --init-from config/crosswalk_roi.json
    python tools/calibrate_roi.py --source data/demo/crosswalk_best_60s.mp4 \
        --risk-scale-from config/crosswalk_roi.json
"""

from pathlib import Path
import argparse
import math
import sys
from datetime import datetime, timezone

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roi_editor import DEFAULT_BEV_SIZE, RoiEditor  # noqa: E402
from src.roi_transform import (  # noqa: E402
    POLYGON_KEYS,
    estimate_risk_scale,
    load_roi_config,
    polygon_area,
    save_roi_config,
    validate_roi_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROI = PROJECT_ROOT / "config" / "webcam_roi.json"

RISK_SCALE_CAVEAT = (
    "NOTE: the pixel thresholds in config.yaml only stay equivalent when the "
    "camera geometry (height, tilt, lens, distance to the crossing) resembles "
    "the reference calibration. Verify the risk levels on site."
)

# An estimate this far from 1.0 usually means the crosswalk polygon covers
# something other than the crossing; the scale is still applied, but the user
# needs telling before a near-zero scale quietly disables every threshold.
IMPLAUSIBLE_SCALE_LOW = 0.25
IMPLAUSIBLE_SCALE_HIGH = 4.0


def parse_source(source):
    """Keep URLs and file paths as strings, but convert camera indices to ints."""
    text = str(source).strip()

    if text.lstrip("-").isdigit():
        return int(text)

    return text


def parse_size(text):
    separator = "x" if "x" in text else ","
    parts = text.lower().split(separator)

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected WxH (for example 1280x720), got '{text}'")

    try:
        width, height = (int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected WxH (for example 1280x720), got '{text}'")

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"width and height must be positive, got '{text}'")

    return width, height


def parse_positive_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got '{text}'")

    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite number, got '{text}'")

    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a ROI calibration onto a camera or video frame.")
    parser.add_argument("--source", type=str, required=True,
                        help="Camera index (0) or video file used to calibrate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROI,
                        help="Where to write the calibration")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Existing ROI JSON to edit; ignored if its frame "
                             "size differs from this source")
    parser.add_argument("--frame-size", type=parse_size, default=None,
                        help="Expected source size as WxH; warns when the "
                             "source delivers something else")
    parser.add_argument("--risk-scale", type=parse_positive_float, default=None,
                        help="Pixel threshold scale; overrides --risk-scale-from")
    parser.add_argument("--risk-scale-from", type=Path, default=None,
                        help="Reference ROI JSON to estimate the risk scale from")
    parser.add_argument("--bev-size", type=parse_size, default=None,
                        help="Bird's-eye panel size as WxH (default 320x220)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Video frame to start from, skipping titles/fades")
    return parser.parse_args()


def open_capture(capture_source):
    cap = cv2.VideoCapture(capture_source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {capture_source}")

    return cap


def read_frame(cap, capture_source, discard=5):
    """Read a frame, skipping the first few which some cameras deliver badly."""
    frame = None

    for _ in range(max(1, discard)):
        ok, frame = cap.read()
        if not ok:
            frame = None

    if frame is None:
        raise RuntimeError(f"Cannot read a frame from: {capture_source}")

    return frame


def build_video_id(capture_source):
    if isinstance(capture_source, int):
        return f"camera_{capture_source}"

    return Path(str(capture_source)).stem


def load_seed(seed_path, frame_size):
    """Read the calibration to edit, or {} when it does not apply here.

    A calibration made for a different frame size describes a different
    geometry, so its polygons are dropped rather than silently reinterpreted
    as if they were drawn for this source.
    """
    seed = load_roi_config(seed_path)
    seed_size = (seed.get("frame_width"), seed.get("frame_height"))

    if seed_size != frame_size:
        print(f"'{seed_path}' is calibrated for {seed_size[0]}x{seed_size[1]}, "
              f"but this source is {frame_size[0]}x{frame_size[1]}.")
        print("Its polygons do not apply here, so the editor starts empty.")
        print("Use tools/make_webcam_roi.py to remap it instead.")
        return {}

    return seed


def seed_bev_size(seed, override):
    if override is not None:
        return override

    homography = seed.get("homography") or {}
    width = homography.get("bev_width")
    height = homography.get("bev_height")

    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height

    return DEFAULT_BEV_SIZE


def estimate_scale_report(reference_path, reference_polygon, target_polygon):
    """Estimate a scale from `reference_path` and describe how it was reached.

    Returns (scale, note) where scale is None when the polygons are too small
    to compare, so the caller can keep whatever it had.
    """
    scale = estimate_risk_scale(reference_polygon, target_polygon)

    if scale is None:
        return None, ("Cannot estimate the risk scale: a crosswalk polygon is "
                      "too small or degenerate.")

    reference_span = math.sqrt(polygon_area(reference_polygon))
    target_span = math.sqrt(polygon_area(target_polygon))

    note = (f"Risk scale estimated from '{reference_path}': "
            f"sqrt(area) {reference_span:.1f} -> {target_span:.1f} pixels.")

    if scale < IMPLAUSIBLE_SCALE_LOW or scale > IMPLAUSIBLE_SCALE_HIGH:
        note += (
            f" That is far from 1.0, so every pixel threshold is scaled by "
            f"{scale}. Check that crosswalk_roi really covers the crossing.")

    return scale, note


def resolve_risk_scale(args, seed_scale, roi_config):
    """Pick the pixel threshold scale, in order of how explicit it was asked for."""
    if args.risk_scale is not None and args.risk_scale_from is not None:
        print("Note: --risk-scale takes precedence over --risk-scale-from.")

    if args.risk_scale is not None:
        return args.risk_scale, "manual"

    if args.risk_scale_from is not None:
        reference_path = args.risk_scale_from.resolve()
        reference = load_roi_config(reference_path)
        reference_polygon = reference.get("crosswalk_roi")
        target_polygon = roi_config.get("crosswalk_roi")

        if not reference_polygon:
            print(f"Warning: '{reference_path}' has no crosswalk_roi to compare.")
        elif not target_polygon:
            print("Warning: draw crosswalk_roi before saving to estimate the "
                  "risk scale.")
        else:
            scale, note = estimate_scale_report(
                reference_path, reference_polygon, target_polygon)

            if scale is not None:
                print(note)
                return scale, f"estimated:{reference_path}"

            print(f"Warning: {note}")

        print("Keeping the previous risk scale.")

    if seed_scale is not None:
        return seed_scale, "init-from"

    return 1.0, "default"


def make_saver(args, output_path, seed_scale, frame_size, video_id, capture_source,
               seed_path):
    """Build the callback the editor calls when the user presses 's'."""
    state = {"risk_scale": 1.0, "origin": "default"}

    def save(roi_config):
        risk_scale, origin = resolve_risk_scale(args, seed_scale, roi_config)

        roi_config["risk_scale"] = risk_scale
        roi_config["risk_scale_source"] = origin
        roi_config["video_id"] = video_id
        roi_config["calibration"] = {
            "tool": "tools/calibrate_roi.py",
            "source": str(capture_source),
            "source_kind": "camera" if isinstance(capture_source, int) else "video",
            "frame_size": f"{frame_size[0]}x{frame_size[1]}",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "init_from": str(seed_path) if seed_path is not None else None,
            "risk_scale_from": (
                str(args.risk_scale_from) if args.risk_scale_from is not None else None),
        }

        # Validate before writing so a calibration the pipeline would reject
        # never reaches disk.
        validate_roi_config(
            roi_config, frame_size[0], frame_size[1], output_path)

        save_roi_config(output_path, roi_config)

        state["risk_scale"] = risk_scale
        state["origin"] = origin

    return save, state


def describe_regions(roi_config):
    return [key for key in POLYGON_KEYS if key in roi_config]


def print_summary(args, roi_config, state, frame_size):
    print()
    print("Written:", args.output)
    print("Frame size:", f"{frame_size[0]}x{frame_size[1]}")
    print("Regions:", ", ".join(describe_regions(roi_config)))

    homography = roi_config.get("homography") or {}
    if homography.get("enabled"):
        print("Homography: enabled "
              f"({homography['bev_width']}x{homography['bev_height']})")
    else:
        print("Homography: disabled")

    origin = state["origin"]
    print(f"Risk scale: {state['risk_scale']} ({origin})")

    if origin.startswith("estimated") or origin == "manual":
        print()
        print(RISK_SCALE_CAVEAT)
    elif origin == "default":
        print()
        print("Nothing scaled the pixel thresholds; they still assume the "
              "camera geometry of config.yaml.")

    print()
    print("Next:")
    print(f"  python run_webcam.py --roi-config {args.output} --source {args.source}")


def main():
    args = parse_args()

    output_path = args.output.resolve()

    seed_path = None
    if args.init_from is not None:
        seed_path = args.init_from.resolve()
        if seed_path == output_path:
            raise SystemExit(
                f"Refusing to overwrite the calibration being edited: {seed_path}")

    capture_source = parse_source(args.source)
    is_camera = isinstance(capture_source, int)
    cap = open_capture(capture_source)

    try:
        # A seek already landed on a decodable frame, so only the initial
        # open needs the warm-up frames a camera or codec delivers badly.
        discard = 5

        if args.start_frame > 0:
            if is_camera:
                print("Note: --start-frame is ignored for a camera source.")
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
                discard = 1

        frame = read_frame(cap, capture_source, discard=discard)
        height, width = frame.shape[:2]
        frame_size = (width, height)
        print(f"Detected source resolution: {width}x{height}")

        if args.frame_size is not None and args.frame_size != frame_size:
            print(f"Warning: --frame-size {args.frame_size[0]}x{args.frame_size[1]} "
                  f"does not match the source ({width}x{height}); the calibration "
                  "records the size the source actually delivers, which is what "
                  "the pipeline checks against.")

        seed = load_seed(seed_path, frame_size) if seed_path is not None else {}
        seed_scale = seed.get("risk_scale")

        polygons = {key: seed[key] for key in POLYGON_KEYS if key in seed}
        bev_width, bev_height = seed_bev_size(seed, args.bev_size)
        homography_seed = dict(seed.get("homography") or {})
        homography_seed["bev_width"] = bev_width
        homography_seed["bev_height"] = bev_height

        video_id = build_video_id(capture_source)

        save, state = make_saver(
            args, output_path, seed_scale, frame_size, video_id, capture_source,
            seed_path)

        print()
        print("Left click adds a point, right click undoes, 's' saves, 'q' quits.")
        print("Use an English keyboard layout: an active IME swallows the keys.")

        editor = RoiEditor(
            frame,
            polygons=polygons,
            homography=homography_seed,
            source_label=str(capture_source),
            capture=cap,
            can_step=not is_camera,
            save_callback=save,
        )

        roi_config = editor.run()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if roi_config is None:
        print("Discarded; nothing written.")
        return

    print_summary(args, roi_config, state, frame_size)


if __name__ == "__main__":
    main()
