from pathlib import Path
import argparse

from src.audio_warning import create_voice_warning_audio, merge_audio_to_video_h264
from src.config import load_config
from src.cv_pipeline import CrosswalkRiskPipeline
from src.locales import DEFAULT_LOCALE, SUPPORTED_LOCALES
from src.report_assets import create_risk_score_timeline, extract_representative_screenshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Run only the CV pipeline without generating voice warning video",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=list(SUPPORTED_LOCALES),
        help="Language for the on-screen text, overrides config `language`",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_config(str(config_path))

    language = args.lang or config.get("language", DEFAULT_LOCALE)
    print(f"Language: {language}")

    pipeline = CrosswalkRiskPipeline(
        config=config, project_root=project_root, language=language)
    result = pipeline.run()

    if args.skip_audio:
        print("Skipped audio generation.")
        return

    paths = config["paths"]
    audio_cfg = config["audio"]

    audio_path = create_voice_warning_audio(
        project_root=project_root,
        output_video_path=str(result["output_video"]),
        event_log_path=str(result["event_log"]),
        audio_output_path=paths["warning_audio"],
        temp_audio_dir=paths["temp_audio_dir"],
        high_text=audio_cfg["high_text"],
        danger_text=audio_cfg["danger_text"],
        high_cooldown_sec=float(audio_cfg["high_cooldown_sec"]),
        danger_cooldown_sec=float(audio_cfg["danger_cooldown_sec"]),
    )

    final_video = merge_audio_to_video_h264(
        project_root=project_root,
        input_video_path=str(result["output_video"]),
        audio_path=str(audio_path),
        output_video_path=paths["output_video_h264"],
    )

    print("Final video with voice warning:", final_video)

    timeline_path = create_risk_score_timeline(
        project_root=project_root,
        frame_log_path=paths["frame_log"],
        output_path=paths["risk_score_timeline"],
        language=language,
    )

    print("Risk score timeline:", timeline_path)

    screenshots = extract_representative_screenshots(
        project_root=project_root,
        final_video_path=paths["output_video_h264"],
        event_log_path=paths["event_log"],
        output_dir="assets/results",
    )

    print("Representative screenshots:")
    for screenshot in screenshots:
        print("-", screenshot)


if __name__ == "__main__":
    main()
