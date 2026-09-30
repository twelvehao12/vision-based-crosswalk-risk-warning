"""Shared ROI calibration helpers: schema, remapping and validation.

A ROI file is calibrated for one specific camera and resolution (recorded in
its ``frame_width``/``frame_height``). To reuse it on a webcam you point at a
screen, the polygons are scaled and offset into the region the screen occupies
in the camera frame, and the result is written to a new file.

The pixel-based risk thresholds travel with the polygons: a smaller image
shrinks both the distances and the boxes, so the calibration's ``risk_scale``
records the uniform scale factor for the pipeline to apply.

This module stays free of heavy imports so the calibration tools can use it
without pulling in the detection pipeline.
"""

from pathlib import Path
import copy
import json
import math


POLYGON_KEYS = (
    "crosswalk_roi",
    "vehicle_approach_zone",
    "pedestrian_waiting_zone",
    "secondary_crosswalk_roi",
    "secondary_vehicle_approach_zone",
)

# Split of POLYGON_KEYS: the three the pipeline requires, and the optional
# pair describing a secondary crossing, which is only valid when both are
# present. Slicing keeps the two views from drifting apart.
REQUIRED_POLYGON_KEYS = POLYGON_KEYS[:3]
SECONDARY_POLYGON_KEYS = POLYGON_KEYS[3:]

# Relative difference below which sx and sy count as one uniform scale.
UNIFORM_SCALE_TOLERANCE = 0.01


def load_roi_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_roi_config(path, roi_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(roi_config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fit_rect(reference_width, reference_height, rect, keep_aspect=True):
    """Fit the reference frame into `rect` as (x, y, width, height).

    Returns the (sx, sy, ox, oy) mapping: x' = x * sx + ox, y' = y * sy + oy.
    With `keep_aspect` the reference aspect ratio is preserved and the content
    is centred inside the rect, so sx == sy and risk scores stay equivalent.
    """
    if len(rect) != 4:
        raise ValueError("rect must be given as X,Y,W,H")

    rect_x, rect_y, rect_w, rect_h = (float(value) for value in rect)

    if rect_w <= 0 or rect_h <= 0:
        raise ValueError(f"rect must have a positive size, got {rect_w}x{rect_h}")

    if keep_aspect:
        scale = min(rect_w / reference_width, rect_h / reference_height)
        sx = sy = scale
        ox = rect_x + (rect_w - reference_width * scale) / 2
        oy = rect_y + (rect_h - reference_height * scale) / 2
    else:
        sx = rect_w / reference_width
        sy = rect_h / reference_height
        ox = rect_x
        oy = rect_y

    return sx, sy, ox, oy


def clamp_point(x, y, frame_width, frame_height):
    """Round a point to whole pixels and clamp it inside the frame.

    Returns `[x, y]` ints within `[0, frame_width - 1] x [0, frame_height - 1]`,
    so a point on the frame edge cannot become the out-of-bounds coordinate the
    pipeline rejects.
    """
    x = min(max(int(round(x)), 0), frame_width - 1)
    y = min(max(int(round(y)), 0), frame_height - 1)

    return [x, y]


def polygon_area(points):
    """Absolute area of a polygon, or 0.0 when it cannot enclose anything."""
    if not points or len(points) < 3:
        return 0.0

    total = 0.0
    count = len(points)

    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        total += x1 * y2 - x2 * y1

    return abs(total) / 2.0


def estimate_risk_scale(reference_polygon, target_polygon):
    """Estimate a risk scale from the pixel size of two matching polygons.

    Pixel distances are linear, so the ratio of the square roots of the areas
    is the dimensionally correct multiplier; a raw area ratio would square the
    thresholds. Returns None when either polygon is too small to measure, so
    the caller can keep whatever value it had.
    """
    reference_area = polygon_area(reference_polygon)
    target_area = polygon_area(target_polygon)

    if reference_area <= 0 or target_area <= 0:
        return None

    return round(math.sqrt(target_area / reference_area), 6)


def _remap_point(point, sx, sy, ox, oy, frame_width, frame_height):
    # Clamping matters because a rect touching the frame edge would otherwise
    # produce the out-of-bounds coordinate the pipeline rejects.
    return clamp_point(
        point[0] * sx + ox, point[1] * sy + oy, frame_width, frame_height)


def remap_roi_config(roi_config, sx, sy, ox, oy, frame_width, frame_height,
                     source_path=None):
    """Return a copy of `roi_config` remapped into a camera frame.

    `frame_width`/`frame_height` are the target camera frame size, which the
    pipeline checks against the frame it actually receives.
    """
    for name, value in (("sx", sx), ("sy", sy), ("ox", ox), ("oy", oy)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if sx <= 0 or sy <= 0:
        raise ValueError(f"scale must be positive, got sx={sx}, sy={sy}")

    remapped = copy.deepcopy(roi_config)

    for key in POLYGON_KEYS:
        if key in remapped:
            remapped[key] = [
                _remap_point(point, sx, sy, ox, oy, frame_width, frame_height)
                for point in remapped[key]
            ]

    homography_cfg = remapped.get("homography")
    if isinstance(homography_cfg, dict) and "src_points" in homography_cfg:
        # bev_width/bev_height describe the bird's-eye output panel, not the
        # camera frame, so they stay as they are.
        homography_cfg["src_points"] = [
            _remap_point(point, sx, sy, ox, oy, frame_width, frame_height)
            for point in homography_cfg["src_points"]
        ]

    remapped["frame_width"] = int(frame_width)
    remapped["frame_height"] = int(frame_height)

    relative_difference = abs(sx - sy) / max(sx, sy)
    if relative_difference <= UNIFORM_SCALE_TOLERANCE:
        uniform_scale = (sx + sy) / 2
        remapped["risk_scale"] = round(uniform_scale, 6)
        uniform = True
    else:
        # A non-uniform stretch has no single pixel scale, so the calibrated
        # risk thresholds cannot be carried over faithfully.
        remapped.pop("risk_scale", None)
        uniform = False

    remapped["remap"] = {
        "source": str(source_path) if source_path is not None else None,
        "reference_width": roi_config.get("frame_width"),
        "reference_height": roi_config.get("frame_height"),
        "sx": round(sx, 6),
        "sy": round(sy, 6),
        "ox": round(ox, 6),
        "oy": round(oy, 6),
        "uniform_scale": uniform,
    }

    return remapped


# --- Validation -----------------------------------------------------------
#
# These mirror the checks the pipeline applies when it loads a ROI file, so a
# calibration tool can prove its output will be accepted before writing it.
# The rules and the wording of the errors are kept identical to
# `CrosswalkRiskPipeline._validate_polygon`, `_validate_roi_structure` and
# `validate_roi_for_frame` in src/cv_pipeline.py.


def validate_roi_polygon(name, points, required_points=None):
    if not isinstance(points, list):
        raise ValueError(f"ROI polygon '{name}' must be a list of points")

    minimum = required_points if required_points is not None else 3
    if len(points) < minimum:
        raise ValueError(
            f"ROI polygon '{name}' needs at least {minimum} points, "
            f"got {len(points)}"
        )

    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"ROI polygon '{name}' point {index} must be an [x, y] pair"
            )

        for value in point:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"ROI polygon '{name}' point {index} has a non-numeric "
                    f"coordinate: {value!r}"
                )
            if not math.isfinite(value):
                raise ValueError(
                    f"ROI polygon '{name}' point {index} has a non-finite "
                    f"coordinate: {value!r}"
                )


def validate_roi_structure(roi_config):
    for name in REQUIRED_POLYGON_KEYS:
        if name not in roi_config:
            raise ValueError(f"ROI config is missing required key '{name}'")
        validate_roi_polygon(name, roi_config[name])

    secondary = [name for name in SECONDARY_POLYGON_KEYS if name in roi_config]
    if len(secondary) == 1:
        raise ValueError(
            f"ROI config defines '{secondary[0]}' without its counterpart; "
            "secondary crosswalk and secondary approach zone must be "
            "provided together"
        )
    for name in secondary:
        validate_roi_polygon(name, roi_config[name])

    for name in ("frame_width", "frame_height"):
        value = roi_config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"ROI config needs a positive integer '{name}' describing "
                "the calibrated frame size"
            )

    risk_scale = roi_config.get("risk_scale", 1.0)
    if isinstance(risk_scale, bool) or not isinstance(risk_scale, (int, float)):
        raise ValueError("ROI config 'risk_scale' must be a number")
    if not math.isfinite(risk_scale) or risk_scale <= 0:
        raise ValueError(
            f"ROI config 'risk_scale' must be a positive finite number, "
            f"got {risk_scale!r}")

    homography_cfg = roi_config.get("homography", {})
    if homography_cfg.get("enabled", False):
        if "src_points" not in homography_cfg:
            raise ValueError(
                "ROI config enables homography but has no 'src_points'")
        validate_roi_polygon(
            "homography.src_points", homography_cfg["src_points"],
            required_points=4)

        for name in ("bev_width", "bev_height"):
            value = homography_cfg.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"ROI config needs a positive integer 'homography.{name}'")


def validate_roi_points_for_frame(roi_config, frame_width, frame_height,
                                  source_path=None):
    """Check that a ROI was calibrated for the frame it is applied to.

    Coordinates are never rescaled implicitly: a ROI calibrated for one camera
    or resolution is meaningless on another, so mismatches fail loudly instead
    of drawing zone outlines in the wrong place.
    """
    calibrated_width = roi_config["frame_width"]
    calibrated_height = roi_config["frame_height"]
    label = source_path if source_path is not None else "<roi config>"

    if (calibrated_width, calibrated_height) != (frame_width, frame_height):
        raise ValueError(
            f"ROI calibration size mismatch for {label}: "
            f"calibrated for {calibrated_width}x{calibrated_height}, but the "
            f"input frame is {frame_width}x{frame_height}. Calibrate a new "
            "ROI for this camera and resolution; coordinates are not scaled "
            "automatically."
        )

    homography_cfg = roi_config.get("homography", {})
    polygon_keys: list[str] = list(POLYGON_KEYS)
    if homography_cfg.get("enabled", False):
        polygon_keys.append("homography.src_points")

    for key in polygon_keys:
        if key == "homography.src_points":
            points = homography_cfg["src_points"]
        elif key in roi_config:
            points = roi_config[key]
        else:
            continue

        for index, (x, y) in enumerate(points):
            if not (0 <= x < frame_width and 0 <= y < frame_height):
                raise ValueError(
                    f"ROI point {index} of '{key}' at ({x}, {y}) is outside "
                    f"the calibrated frame {frame_width}x{frame_height} "
                    f"({label})"
                )


def validate_roi_config(roi_config, frame_width=None, frame_height=None,
                        source_path=None):
    """Run every check the pipeline applies, optionally including frame size.

    Pass `frame_width`/`frame_height` to also assert the calibration matches a
    specific capture size.
    """
    validate_roi_structure(roi_config)

    if frame_width is not None and frame_height is not None:
        validate_roi_points_for_frame(
            roi_config, frame_width, frame_height, source_path)
