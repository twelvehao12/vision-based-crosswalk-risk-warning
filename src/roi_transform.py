"""Remap a ROI calibration onto a different camera frame.

A ROI file is calibrated for one specific camera and resolution (recorded in
its ``frame_width``/``frame_height``). To reuse it on a webcam you point at a
screen, the polygons are scaled and offset into the region the screen occupies
in the camera frame, and the result is written to a new file.

The pixel-based risk thresholds travel with the polygons: a smaller image
shrinks both the distances and the boxes, so the calibration's ``risk_scale``
records the uniform scale factor for the pipeline to apply.
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


def _remap_point(point, sx, sy, ox, oy, frame_width, frame_height):
    x = point[0] * sx + ox
    y = point[1] * sy + oy

    # Round to whole pixels, then clamp so a rect touching the frame edge
    # cannot produce the out-of-bounds coordinate the pipeline rejects.
    x = min(max(int(round(x)), 0), frame_width - 1)
    y = min(max(int(round(y)), 0), frame_height - 1)

    return [x, y]


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
