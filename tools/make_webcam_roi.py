"""Calibrate a webcam ROI file from the camera-authored calibration.

The bundled `config/crosswalk_roi.json` is calibrated for the 1200x1100 demo
video. To point a camera at a screen playing that video, the polygons have to
be moved into the region the screen occupies in the camera frame. This tool
does that and writes a new file (by default `config/webcam_roi.json`); the
source calibration is never modified.

Usage:
    python tools/make_webcam_roi.py --source 0 --interactive
    python tools/make_webcam_roi.py --source 0 --rect 120,80,640,590
"""

from pathlib import Path
import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.roi_transform import (  # noqa: E402
    POLYGON_KEYS,
    fit_rect,
    load_roi_config,
    remap_roi_config,
    save_roi_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_ROI = PROJECT_ROOT / "config" / "crosswalk_roi.json"
DEFAULT_OUTPUT_ROI = PROJECT_ROOT / "config" / "webcam_roi.json"

DEFAULT_REFERENCE_SIZE = (1200, 1100)

POLYGON_COLORS = {
    "crosswalk_roi": (0, 255, 255),
    "vehicle_approach_zone": (0, 0, 255),
    "pedestrian_waiting_zone": (255, 0, 0),
    "secondary_crosswalk_roi": (0, 255, 0),
    "secondary_vehicle_approach_zone": (255, 0, 255),
}


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


def parse_rect(text):
    parts = text.split(",")

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"expected X,Y,W,H (for example 120,80,640,590), got '{text}'")

    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected X,Y,W,H (for example 120,80,640,590), got '{text}'")


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


def resolve_reference_size(roi_config, override):
    if override is not None:
        return override

    width = roi_config.get("frame_width")
    height = roi_config.get("frame_height")

    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height

    print(
        f"Source ROI has no frame_width/frame_height; "
        f"assuming {DEFAULT_REFERENCE_SIZE[0]}x{DEFAULT_REFERENCE_SIZE[1]}.")

    return DEFAULT_REFERENCE_SIZE


def resolve_frame_size(args, cap, capture_source):
    if args.frame_size is not None:
        return args.frame_size

    if cap is None:
        raise RuntimeError(
            "Cannot detect the frame size without a source; pass --frame-size WxH")

    frame = read_frame(cap, capture_source)
    height, width = frame.shape[:2]
    print(f"Detected source resolution: {width}x{height}")

    return width, height


def resolve_rect(args, frame_size, cap, capture_source):
    if args.rect is not None:
        return args.rect

    if args.interactive:
        return select_rect(cap, capture_source)

    # No rect given and not interactive: use the whole frame and let
    # fit_rect centre the reference aspect ratio inside it.
    width, height = frame_size
    print(f"No --rect given; using the whole frame ({width}x{height}) as the screen area.")

    return 0, 0, width, height


def select_rect(cap, capture_source):
    frame = read_frame(cap, capture_source, discard=10)

    print("Drag a box around the video on the screen, then press ENTER/SPACE.")

    rect = cv2.selectROI("Select the screen area", frame, showCrosshair=False)
    cv2.destroyWindow("Select the screen area")

    if rect[2] == 0 or rect[3] == 0:
        raise RuntimeError("No screen area selected")

    return tuple(int(value) for value in rect)


def build_remapped(roi_config, reference_size, frame_size, rect, keep_aspect,
                   source_path=DEFAULT_SOURCE_ROI):
    sx, sy, ox, oy = fit_rect(
        reference_size[0], reference_size[1], rect, keep_aspect=keep_aspect)

    return remap_roi_config(
        roi_config,
        sx=sx,
        sy=sy,
        ox=ox,
        oy=oy,
        frame_width=frame_size[0],
        frame_height=frame_size[1],
        source_path=source_path,
    )


def draw_overlay(frame, roi_config):
    for key in POLYGON_KEYS:
        if key not in roi_config:
            continue

        points = roi_config[key]
        color = POLYGON_COLORS.get(key, (255, 255, 255))

        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, color, 2)
        cv2.putText(
            frame,
            key,
            (int(points[0][0]), max(15, int(points[0][1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return frame


def preview(cap, capture_source, source_roi_config, remapped, reference_size,
            frame_size, initial_rect, keep_aspect):
    """Live preview so the ROI can be checked against the real scene.

    Returns the accepted (remapped_config, rect), or None when discarded.
    """
    rect = initial_rect

    while True:
        frame = read_frame(cap, capture_source, discard=1)
        frame = draw_overlay(frame, remapped)

        cv2.putText(
            frame,
            "s: save   r: re-select screen   q: quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Webcam ROI preview", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return None

        if key == ord("s"):
            cv2.destroyAllWindows()
            return remapped, rect

        if key == ord("r"):
            cv2.destroyAllWindows()
            rect = select_rect(cap, capture_source)
            remapped = build_remapped(
                source_roi_config,
                reference_size,
                frame_size,
                rect,
                keep_aspect,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build a camera ROI file by remapping an existing calibration.")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index or video/stream used to size the frame")
    parser.add_argument("--source-roi", type=Path, default=DEFAULT_SOURCE_ROI,
                        help="Calibration to remap (never modified)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROI,
                        help="Where to write the remapped calibration")
    parser.add_argument("--rect", type=parse_rect, default=None,
                        help="Screen area in the camera frame as X,Y,W,H")
    parser.add_argument("--interactive", action="store_true",
                        help="Pick the screen area with the mouse and preview live")
    parser.add_argument("--frame-size", type=parse_size, default=None,
                        help="Target camera frame size as WxH; detected from --source if omitted")
    parser.add_argument("--reference-size", type=parse_size, default=None,
                        help="Calibrated frame size of --source-roi; read from it if omitted")
    parser.add_argument("--no-keep-aspect", dest="keep_aspect", action="store_false",
                        help="Stretch the calibration instead of preserving its aspect ratio")
    parser.set_defaults(keep_aspect=True)

    args = parser.parse_args()

    source_roi_path = args.source_roi.resolve()
    output_path = args.output.resolve()

    if source_roi_path == output_path:
        raise SystemExit(
            f"Refusing to overwrite the source calibration: {source_roi_path}")

    roi_config = load_roi_config(source_roi_path)
    reference_size = resolve_reference_size(roi_config, args.reference_size)

    capture_source = parse_source(args.source)

    # A capture is only needed to detect the frame size or to run the
    # interactive picker, so a fully specified rect stays off-camera.
    cap = None
    if args.frame_size is None or args.interactive:
        cap = open_capture(capture_source)

    try:
        frame_size = resolve_frame_size(args, cap, capture_source)
        rect = resolve_rect(args, frame_size, cap, capture_source)

        remapped = build_remapped(
            roi_config, reference_size, frame_size, rect, args.keep_aspect,
            source_path=source_roi_path)

        if not args.interactive:
            save_roi_config(output_path, remapped)
        else:
            saved = preview(
                cap, capture_source, roi_config, remapped, reference_size,
                frame_size, rect, args.keep_aspect)

            if saved is None:
                print("Discarded; nothing written.")
                return

            remapped, rect = saved
            save_roi_config(output_path, remapped)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    scale = remapped.get("risk_scale")

    print()
    print("Source calibration:", source_roi_path)
    print("Written:", output_path)
    print("Calibrated size:", f"{remapped['frame_width']}x{remapped['frame_height']}")
    print("Screen area:", f"x={rect[0]} y={rect[1]} w={rect[2]} h={rect[3]}")

    if scale is None:
        print("Risk scale: not set (non-uniform stretch); pixel thresholds are")
        print("no longer equivalent to the source calibration.")
    else:
        print(f"Risk scale: {scale} (pixel thresholds scaled by this factor)")

    print()
    print("Next:")
    print(f"  python run_webcam.py --roi-config {args.output} --source {args.source}")


if __name__ == "__main__":
    main()
