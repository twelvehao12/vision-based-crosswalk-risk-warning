"""Real-time webcam risk warning.

This is a thin front-end over `CrosswalkRiskPipeline`: it only picks the
capture source and the ROI calibration, then hands both to the same per-frame
pipeline `main.py` uses, so detection, risk scoring, ROI overlays, the
bird's-eye panel and the warning banner all behave identically.

A webcam sees a different frame than the calibrated video, so it needs its own
ROI file. Build one with:

    python tools/make_webcam_roi.py --source 0 --interactive

Then run:

    python run_webcam.py --roi-config config/webcam_roi.json --source 0
"""

from pathlib import Path
import argparse
import copy

from src.config import load_config
from src.cv_pipeline import CrosswalkRiskPipeline
from src.locales import DEFAULT_LOCALE, SUPPORTED_LOCALES


DEFAULT_CONFIG = "config/config.yaml"


def parse_source(source):
    """Keep URLs and file paths as strings, but convert camera indices to ints."""
    text = str(source).strip()

    if text.lstrip("-").isdigit():
        return int(text)

    return text


def build_webcam_config(config, args):
    """Build an isolated pipeline config using CLI > webcam > shared values."""
    runtime_config = copy.deepcopy(config)

    paths = runtime_config.setdefault("paths", {})
    model_cfg = runtime_config.setdefault("model", {})
    webcam_cfg = runtime_config.get("webcam", {})

    project_root = Path(__file__).resolve().parent

    roi_config = (
        args.roi_config or webcam_cfg.get("roi_config") or paths["roi_config"])

    # The camera calibration is built by a separate tool, so a missing file
    # should explain itself rather than fail deep inside the pipeline.
    if args.roi_config is None and "roi_config" in webcam_cfg:
        if not (project_root / roi_config).exists():
            print(f"Webcam ROI '{roi_config}' not found; falling back to "
                  f"'{paths['roi_config']}'.")
            print("Run 'python tools/make_webcam_roi.py --source 0 "
                  "--interactive' to calibrate this camera.")
            roi_config = paths["roi_config"]

    paths["roi_config"] = roi_config
    paths["output_video"] = (
        args.output or webcam_cfg.get("output_video") or paths["output_video"])
    paths["event_log"] = webcam_cfg.get(
        "event_log", "outputs/logs/webcam_event_log.csv")
    paths["frame_log"] = webcam_cfg.get(
        "frame_log", "outputs/logs/webcam_frame_log.csv")
    paths["screenshot_dir"] = webcam_cfg.get(
        "screenshot_dir", "outputs/screenshots/webcam")

    if args.model is not None:
        model_cfg["name"] = args.model
    if args.conf is not None:
        model_cfg["confidence_threshold"] = args.conf
    if args.imgsz is not None:
        model_cfg["image_size"] = args.imgsz
    if args.device is not None:
        model_cfg["device"] = args.device

    language = args.lang or runtime_config.get("language", DEFAULT_LOCALE)

    source = parse_source(
        args.source if args.source is not None else webcam_cfg.get("source", 0))

    show_window = bool(webcam_cfg.get("show_window", True))

    return runtime_config, source, language, show_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to config YAML file")
    parser.add_argument("--roi-config", type=str, default=None,
                        help="Path to camera-specific ROI JSON; overrides config")
    parser.add_argument("--source", type=str, default=None,
                        help="Webcam index or video/RTSP source; overrides config")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--save", action="store_true",
                        help="Save webcam output video")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--lang", type=str, default=None,
                        choices=list(SUPPORTED_LOCALES),
                        help="Language for the on-screen text; overrides config")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(config_path)
    runtime_config, source, language, show_window = build_webcam_config(
        config, args)

    print(f"Language: {language}")
    print("Webcam/video source:", source)
    print("ROI config:", runtime_config["paths"]["roi_config"])

    if show_window:
        print("Press 'q' to quit.")

    pipeline = CrosswalkRiskPipeline(
        config=runtime_config, project_root=project_root, language=language)

    result = pipeline.run(
        source=source, show_window=show_window, save_video=args.save)

    if result["output_video"] is not None:
        print("Output video:", result["output_video"])


if __name__ == "__main__":
    main()
