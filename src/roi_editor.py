"""Interactive ROI calibration editor.

Lets a user draw the polygons the pipeline expects straight onto a camera or
video frame, plus the optional homography corner points, and hands back a
calibration config.

The editor only produces geometry: the caller decides where the result goes
and runs `roi_transform.validate_roi_config` on it before writing, so this
module never needs to know the output schema rules.
"""

import numpy as np
import cv2

from src.roi_transform import (
    POLYGON_KEYS,
    REQUIRED_POLYGON_KEYS,
    SECONDARY_POLYGON_KEYS,
    clamp_point,
)


WINDOW_NAME = "ROI calibration"

POLYGON_COLORS = {
    "crosswalk_roi": (0, 255, 255),
    "vehicle_approach_zone": (0, 0, 255),
    "pedestrian_waiting_zone": (255, 0, 0),
    "secondary_crosswalk_roi": (0, 255, 0),
    "secondary_vehicle_approach_zone": (255, 0, 255),
}

POLYGON_HINTS = {
    "crosswalk_roi": "the crossing surface pedestrians walk on",
    "vehicle_approach_zone": "the lane vehicles approach the crossing from",
    "pedestrian_waiting_zone": "the kerb area where pedestrians wait",
    "secondary_crosswalk_roi": "the second crossing, if the junction has one",
    "secondary_vehicle_approach_zone": "the approach lane for the second crossing",
}

# Click order for the road plane, matching the destination corners in
# `setup_homography`, so the bird's-eye view comes out upright.
HOMOGRAPHY_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")

DEFAULT_BEV_SIZE = (320, 220)

TEXT_COLOR = (235, 235, 235)
DIM_COLOR = (150, 150, 150)
INFO_COLOR = (0, 255, 255)
ERROR_COLOR = (0, 0, 255)
OK_COLOR = (0, 200, 0)

HUD_LINE_HEIGHT = 24
HUD_PADDING = 12
STATUS_TTL = 250


def _dashed_line(canvas, start, end, color, dash=9, gap=7, thickness=1):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))

    if length <= 0:
        return

    direction = (end - start) / length
    position = 0.0

    while position < length:
        head = np.round(start + direction * position).astype(int)
        tail = np.round(
            start + direction * min(position + dash, length)).astype(int)
        cv2.line(canvas, tuple(head), tuple(tail), color, thickness)
        position += dash + gap


def _dim(color, factor=0.55):
    return tuple(int(channel * factor) for channel in color)


class RoiEditor:
    """Draw ROI polygons and homography points on a frame.

    The frame is frozen while editing: a live camera would move the scene out
    from under each click. `f` (or space) toggles back to live for a look at
    the real traffic.
    """

    def __init__(self, frame, polygons=None, homography=None, source_label="",
                 capture=None, can_step=False, save_callback=None):
        self.frame = frame
        self.frame_height, self.frame_width = frame.shape[:2]

        self.polygons = {
            key: [list(point) for point in points]
            for key, points in (polygons or {}).items()
            if key in POLYGON_KEYS
        }

        # Secondaries are all-or-nothing, so a seeded config that has them
        # turns the pair on; one without them leaves them off.
        self.secondaries_enabled = any(
            self.polygons.get(key) for key in SECONDARY_POLYGON_KEYS)
        self.secondaries_decided = self.secondaries_enabled

        self.active_key = REQUIRED_POLYGON_KEYS[0]

        self.homography = {
            "enabled": bool(homography and homography.get("enabled")),
            "src_points": [
                list(point) for point in (homography or {}).get("src_points", [])
            ],
            "bev_width": int((homography or {}).get("bev_width", DEFAULT_BEV_SIZE[0])),
            "bev_height": int((homography or {}).get("bev_height", DEFAULT_BEV_SIZE[1])),
        }

        self.homography_capture = None
        self.cursor = None
        self.frozen = True
        self.overlay_visible = True
        self.hud_visible = True
        self.prompt = None
        self.status = None
        self.status_error = False
        self.status_ttl = 0
        self.dirty = False

        self.source_label = source_label
        self.capture = capture
        self.can_step = can_step
        self.save_callback = save_callback

    # --- public ----------------------------------------------------------

    def run(self):
        """Show the editor until the user saves or gives up.

        Returns the calibration config, or None when nothing was saved.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse, self)

        try:
            while True:
                self._refresh_live_frame()
                cv2.imshow(WINDOW_NAME, self._compose())

                # 20 ms rather than 1: the Windows backend dispatches mouse
                # events far more reliably with a slightly longer wait.
                key = cv2.waitKey(20) & 0xFF

                if self._window_closed():
                    break

                if key != 255:
                    action = self._handle_key(key)

                    if action == "quit":
                        break
                    if action == "save":
                        saved = self._save()
                        if saved is not None:
                            return saved

                self._tick_status()
        finally:
            cv2.destroyAllWindows()

        return None

    def build_config(self):
        """Assemble the geometry the caller can validate and write out."""
        config = {
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
        }

        for key in POLYGON_KEYS:
            points = self.polygons.get(key)
            if points:
                config[key] = [list(point) for point in points]

        if self.homography["enabled"]:
            config["homography"] = {
                "enabled": True,
                "src_points": [list(p) for p in self.homography["src_points"]],
                "bev_width": int(self.homography["bev_width"]),
                "bev_height": int(self.homography["bev_height"]),
            }

        return config

    # --- state helpers ---------------------------------------------------

    def _enable_secondary_keys(self):
        self.secondaries_enabled = True
        self.secondaries_decided = True
        for key in SECONDARY_POLYGON_KEYS:
            self.polygons.setdefault(key, [])

    def _disable_secondary_keys(self):
        self.secondaries_enabled = False
        self.secondaries_decided = True
        for key in SECONDARY_POLYGON_KEYS:
            self.polygons.pop(key, None)

    def _select(self, key):
        if key in SECONDARY_POLYGON_KEYS:
            self._enable_secondary_keys()
        self.active_key = key
        self._set_status(f"Active region: {key}")

    def _cycle(self, step):
        steps = list(REQUIRED_POLYGON_KEYS)
        if self.secondaries_enabled:
            steps += list(SECONDARY_POLYGON_KEYS)

        index = steps.index(self.active_key) if self.active_key in steps else 0
        self._select(steps[(index + step) % len(steps)])

    def _set_status(self, text, error=False, ttl=STATUS_TTL):
        self.status = text
        self.status_error = error
        self.status_ttl = ttl

    def _tick_status(self):
        if self.status_ttl > 0:
            self.status_ttl -= 1
            if self.status_ttl == 0:
                self.status = None

    def _ask(self, text, on_yes, on_no=None):
        self.prompt = {"text": text, "on_yes": on_yes, "on_no": on_no}

    def _refresh_live_frame(self):
        if self.frozen or self.capture is None:
            return

        ok, frame = self.capture.read()

        if not ok or frame is None:
            self.frozen = True
            self._set_status(
                "Cannot read from the source any more; staying frozen.", error=True)
            return

        height, width = frame.shape[:2]
        if (width, height) != (self.frame_width, self.frame_height):
            # The calibration is only valid for one exact frame size.
            self.frozen = True
            self._set_status(
                f"Source switched to {width}x{height}; the calibration is for "
                f"{self.frame_width}x{self.frame_height}. Restart with the "
                "right resolution.", error=True, ttl=0)
            return

        self.frame = frame

    def _step_frame(self, delta):
        if self.capture is None or not self.can_step:
            self._set_status("Frame stepping needs a video file source.")
            return

        self.frozen = True
        current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        target = max(0, current + delta)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.capture.read()

        if not ok or frame is None:
            self._set_status("Cannot read that frame.", error=True)
            return

        self.frame = frame
        self._set_status(f"Frame {target}")

    def _window_closed(self):
        try:
            return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # --- mouse -----------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if self.prompt is not None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.homography_capture is not None:
                self._add_homography_point(x, y)
            else:
                self._add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._undo()

    def _add_point(self, x, y):
        point = clamp_point(x, y, self.frame_width, self.frame_height)
        self.polygons.setdefault(self.active_key, []).append(point)
        self.dirty = True

    def _undo(self):
        if self.homography_capture is not None:
            points = self.homography["src_points"]
            if points:
                points.pop()
                self.homography_capture = len(points)
            return

        points = self.polygons.get(self.active_key)

        if points:
            points.pop()
            self.dirty = True

    def _add_homography_point(self, x, y):
        point = clamp_point(x, y, self.frame_width, self.frame_height)
        points = self.homography["src_points"]

        if len(points) >= 4:
            return

        points.append(point)
        self.homography_capture = len(points)
        self.dirty = True

        if len(points) == 4:
            self.homography_capture = None
            self.homography["enabled"] = True
            self._set_status(
                "Homography points captured (TL, TR, BR, BL). Press 'H' to redo.")

    # --- keyboard --------------------------------------------------------

    def _handle_key(self, key):
        if self.prompt is not None:
            return self._handle_prompt_key(key)

        if ord("1") <= key <= ord("5"):
            self._select(POLYGON_KEYS[key - ord("1")])
            return "continue"

        if key == 9:
            self._cycle(1)
            return "continue"
        if key == 25:  # Shift+Tab where the backend reports it
            self._cycle(-1)
            return "continue"

        if key in (ord("u"),):
            self._undo()
        elif key == ord("c"):
            self.polygons[self.active_key] = []
            self.dirty = True
            self._set_status(f"Cleared {self.active_key}")
        elif key == ord("x"):
            self._ask("Clear every region and start over? (y/n)", self._reset_all)
        elif key == ord("n"):
            self._advance()
        elif key in (ord("f"), ord(" ")):
            self._toggle_freeze()
        elif key == ord("o"):
            self.overlay_visible = not self.overlay_visible
        elif key == ord("h"):
            self.hud_visible = not self.hud_visible
        elif key == ord("H"):
            self._begin_homography()
        elif key == ord("e"):
            self.homography["enabled"] = not self.homography["enabled"]
            self._set_status(
                f"Homography {'enabled' if self.homography['enabled'] else 'disabled'}")
        elif key == ord("s"):
            return "save"
        elif key in (ord("q"), 27):
            return self._request_quit()
        elif key == ord(","):
            self._step_frame(-1)
        elif key == ord("."):
            self._step_frame(1)

        return "continue"

    def _handle_prompt_key(self, key):
        prompt = self.prompt

        if key == ord("y"):
            self.prompt = None
            if prompt["on_yes"] is not None:
                return prompt["on_yes"]() or "continue"
        elif key in (ord("n"), 27):
            self.prompt = None
            if prompt["on_no"] is not None:
                return prompt["on_no"]() or "continue"
            self._set_status("Cancelled.")
        else:
            self._set_status("Answer with 'y' or 'n'.", error=True)

        return "continue"

    def _toggle_freeze(self):
        if self.capture is None:
            self._set_status("This source has no live feed; staying frozen.")
            return

        self.frozen = not self.frozen
        self._set_status("Frozen" if self.frozen else "Live preview")

    def _begin_homography(self):
        self.homography["src_points"] = []
        self.homography_capture = 0
        self.homography["enabled"] = True
        self.dirty = True
        self._set_status(
            "Click the four road-plane corners in order: "
            + ", ".join(HOMOGRAPHY_LABELS))

    def _reset_all(self):
        self.polygons = {}
        self.secondaries_enabled = False
        self.secondaries_decided = False
        self.active_key = REQUIRED_POLYGON_KEYS[0]
        self.homography["src_points"] = []
        self.homography["enabled"] = False
        self.homography_capture = None
        self.dirty = False
        self._set_status("All regions cleared.")

    def _advance(self):
        steps = list(REQUIRED_POLYGON_KEYS)
        if self.secondaries_enabled:
            steps += list(SECONDARY_POLYGON_KEYS)

        if self.active_key in steps:
            index = steps.index(self.active_key)
            if index + 1 < len(steps):
                self._select(steps[index + 1])
                return

        if not self.secondaries_decided:
            self._ask(
                "Does this junction have a second crossing to calibrate? (y/n)",
                self._enable_secondary_keys_flow, self._skip_secondaries)
            return

        self._set_status(
            "All regions are drawn - press 's' to save, or 1-5 to revise one.")

    def _enable_secondary_keys_flow(self):
        self._enable_secondary_keys()
        self._select(SECONDARY_POLYGON_KEYS[0])
        self._set_status("Draw the second crossing, then 'n' for its approach lane.")

    def _skip_secondaries(self):
        self._disable_secondary_keys()
        self._set_status("No second crossing; press 's' to save.")

    def _request_quit(self):
        if not self.dirty:
            return "quit"
        self._ask("Discard the unsaved calibration? (y/n)", self._confirm_quit)
        return "continue"

    def _confirm_quit(self):
        return "quit"

    # --- saving ----------------------------------------------------------

    def _save(self):
        config = self.build_config()

        try:
            if self.save_callback is not None:
                self.save_callback(config)
        except ValueError as error:
            # Stay in the editor: the user can fix the geometry and retry.
            self._set_status(str(error), error=True)
            print(f"Not saved: {error}")
            return None

        self.dirty = False
        return config

    # --- rendering -------------------------------------------------------

    def _compose(self):
        canvas = self.frame.copy()

        if self.overlay_visible:
            self._draw_polygons(canvas)
            self._draw_homography(canvas)

        if self.cursor is not None:
            self._draw_cursor(canvas)

        if self.hud_visible:
            self._draw_hud(canvas)

        return canvas

    def _draw_polygons(self, canvas):
        for key in POLYGON_KEYS:
            points = self.polygons.get(key)

            if not points:
                continue

            active = key == self.active_key and self.homography_capture is None
            color = POLYGON_COLORS.get(key, (255, 255, 255))
            if not active:
                color = _dim(color)

            thickness = 3 if active else 2
            array = np.array(points, dtype=np.int32)

            if len(array) >= 3:
                cv2.polylines(canvas, [array], True, color, thickness)
            elif len(array) == 2:
                cv2.line(canvas, tuple(array[0]), tuple(array[1]), color, thickness)

            for point in points:
                cv2.circle(canvas, tuple(point), 4 if active else 3, color, -1)

            cv2.putText(
                canvas,
                key,
                (int(points[0][0]), max(15, int(points[0][1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def _draw_homography(self, canvas):
        points = self.homography["src_points"]

        if not points:
            return

        color = (255, 255, 255) if self.homography["enabled"] else DIM_COLOR

        if len(points) >= 3:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], True, color, 2)

        for index, point in enumerate(points):
            label = HOMOGRAPHY_LABELS[index] if index < len(HOMOGRAPHY_LABELS) else str(index)
            cv2.drawMarker(
                canvas, tuple(point), color, cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(
                canvas,
                f"H{index + 1} {label}",
                (int(point[0]) + 10, int(point[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    def _draw_cursor(self, canvas):
        x, y = self.cursor

        # The rubber band makes it much easier to land the next vertex.
        if self.homography_capture is not None:
            anchor = (
                self.homography["src_points"][-1]
                if self.homography["src_points"] else None)
        else:
            points = self.polygons.get(self.active_key) or []
            anchor = points[-1] if points else None

        if anchor is not None:
            _dashed_line(canvas, anchor, (x, y), INFO_COLOR)

        cv2.putText(
            canvas,
            f"({x}, {y})",
            (x + 12, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            INFO_COLOR,
            1,
        )

    def _hud_lines(self):
        mode = "FROZEN" if self.frozen else "LIVE"
        lines = [
            (f"ROI CALIBRATION   source={self.source_label}   "
             f"mode={mode}   frame={self.frame_width}x{self.frame_height}",
             TEXT_COLOR),
        ]

        if self.homography_capture is not None:
            index = self.homography_capture
            lines.append((
                f"Click homography point {index + 1}/4: {HOMOGRAPHY_LABELS[index]}",
                INFO_COLOR))
        else:
            points = self.polygons.get(self.active_key) or []
            index = POLYGON_KEYS.index(self.active_key) + 1
            lines.append((
                f"Active: {index} {self.active_key}   points={len(points)}   "
                f"- {POLYGON_HINTS.get(self.active_key, '')}",
                POLYGON_COLORS.get(self.active_key, TEXT_COLOR)))

        lines.append(("", TEXT_COLOR))

        for index, key in enumerate(POLYGON_KEYS, start=1):
            lines.append(self._region_line(index, key))

        lines.append(("", TEXT_COLOR))
        lines.append(
            ("LMB add   RMB/u undo   c clear   x reset all   n next region",
             DIM_COLOR))
        lines.append(
            ("1-5 / Tab select   f or space freeze/live   o overlay   h hud   "
             ", . step frame", DIM_COLOR))
        lines.append(
            ("H homography points   e toggle homography   s save   q quit",
             DIM_COLOR))

        if self.homography["enabled"]:
            lines.append((
                f"Homography: on   bev={self.homography['bev_width']}"
                f"x{self.homography['bev_height']}   "
                f"points={len(self.homography['src_points'])}/4",
                TEXT_COLOR))

        return lines

    def _region_line(self, index, key):
        points = self.polygons.get(key)
        used = bool(points)

        if key in SECONDARY_POLYGON_KEYS and not self.secondaries_enabled:
            return (f"[-] {index} {key} (not used)", DIM_COLOR)

        if key == self.active_key:
            marker = ">"
        elif used and len(points) >= 3:
            marker = "x"
        else:
            marker = " "

        color = OK_COLOR if used and len(points) >= 3 else TEXT_COLOR
        if key == self.active_key:
            color = POLYGON_COLORS.get(key, TEXT_COLOR)

        summary = f"({len(points)} points)" if used else "(empty)"

        return (f"[{marker}] {index} {key} {summary}", color)

    def _draw_hud(self, canvas):
        lines = self._hud_lines()

        if self.prompt is not None:
            lines.append(("", TEXT_COLOR))
            lines.append((self.prompt["text"], INFO_COLOR))

        if self.status:
            lines.append(("", TEXT_COLOR))
            lines.append(
                (self.status, ERROR_COLOR if self.status_error else INFO_COLOR))

        height = HUD_PADDING * 2 + HUD_LINE_HEIGHT * len(lines)
        height = min(height, canvas.shape[0])
        banner = canvas[0:height, 0:canvas.shape[1]]
        overlay = banner.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, banner, 0.3, 0, banner)

        baseline = HUD_PADDING + HUD_LINE_HEIGHT - 8

        for index, (text, color) in enumerate(lines):
            if not text:
                continue
            cv2.putText(
                canvas,
                text,
                (HUD_PADDING, baseline + index * HUD_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
            )
