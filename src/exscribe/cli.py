import argparse
import json
import logging
import os
from pathlib import Path

from .frame_processing import get_unique_frames, apply_edge_detection
from .models import SubtitleEntry
from .subtitle import convert_to_srt, merge_subtitle_frames, post_process
from .transcription import transcribe_frames


def process_video(video_path: str, args: argparse.Namespace):
    """Processes a single video file and generates subtitles."""
    logging.info(f"Processing video: {video_path}")

    # Determine cache file path based on debug mode
    output_file = os.path.splitext(video_path)[0] + ".srt"
    cache_file = None

    if args.debug:
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)

        # Create cache file path inside debug directory
        cache_file = debug_dir / (
            os.path.splitext(os.path.basename(video_path))[0] + "_transcriptions.json"
        )

    merged_subtitles = []

    if cache_file and cache_file.exists():
        with open(cache_file, encoding="utf8") as f:
            try:
                merged_subtitles = [
                    SubtitleEntry.from_json(entry) for entry in json.load(f)
                ]
                logging.info(
                    f"Loaded {len(merged_subtitles)} cached subtitle entries from {cache_file}."
                )
            except (json.JSONDecodeError, KeyError):
                logging.warning(
                    f"Failed to load cached subtitle entries from {cache_file}. Processing from scratch."
                )
                pass

    if not merged_subtitles:
        # Extract unique frames
        unique_frames = get_unique_frames(
            video_path, args.similarity_threshold, args.frame_skip, args.scale
        )

        # Apply edge detection if specified
        if args.edge_detection:
            unique_frames = apply_edge_detection(unique_frames)

        # Transcribe frames using selected provider API
        api_key = getattr(args, f"{args.provider}_api_key")
        model_name = getattr(args, f"{args.provider}_model", args.model_name)

        unique_frames = transcribe_frames(
            unique_frames,
            provider_name=args.provider,
            api_key=api_key,
            model_name=model_name,
            language=args.language,
            retry_delay=args.retry_delay,
            max_retries=args.max_retries,
            batch_size=args.batch_size,
            cache_file=cache_file,
            save_every_n=args.save_every_n
            if args.debug
            else 0,  # Only save cache if debug is enabled
        )

        # Merge subtitle frames
        merged_subtitles = merge_subtitle_frames(unique_frames)

        if args.debug and cache_file:
            with open(cache_file, "w", encoding="utf8") as f:
                json.dump(
                    [entry.to_json() for entry in merged_subtitles],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logging.info(f"Saved subtitle entries to cache file: {cache_file}")

    # Post-process subtitles
    merged_subtitles = post_process(
        merged_subtitles, extend_prev=args.extend_prev, extend_next=args.extend_next
    )

    # Write SRT to file
    try:
        # Convert to SRT format
        srt = convert_to_srt(merged_subtitles)
        with open(output_file, "w") as f:
            f.write(srt)
        logging.info(f"Subtitles saved to: {output_file}")
    except Exception as e:
        logging.error(f"Error writing SRT file: {e}")


def main():
    """Main function to process video and generate subtitles."""
    parser = argparse.ArgumentParser(
        description="Transcribe video subtitles using AI providers."
    )

    # Main arguments
    parser.add_argument("input_path", help="Path to the video file.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save intermediate files and additional information.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help="Similarity threshold for frame comparison. Set to 0 to disable.",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=10,
        help="Number of frames to skip during processing.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor for resizing frames."
    )
    parser.add_argument("--output_file", help="Output SRT file path.")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--edge_detection", action="store_true", help="Apply edge detection to frames."
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=60,
        help="Delay in seconds before retrying API calls.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of frames to process in each batch.",
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=10,
        help="Save transcriptions to cache file every n batches.",
    )
    parser.add_argument(
        "--language", type=str, default="English", help="Language of the subtitles."
    )
    parser.add_argument(
        "--extend_prev",
        type=float,
        default=0.2,
        help="Extend previous subtitle entry by this many milliseconds.",
    )
    parser.add_argument(
        "--extend_next",
        type=float,
        default=0.2,
        help="Extend next subtitle entry by this many milliseconds.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai"],
        help="Provider for transcription API (gemini, openai).",
    )

    # Create provider-specific argument groups
    gemini_group = parser.add_argument_group("Gemini Provider Options")
    gemini_group.add_argument(
        "--gemini_api_key",
        help="Gemini API key (can also set GEMINI_API_KEY env variable).",
        default=os.environ.get("GEMINI_API_KEY"),
    )
    gemini_group.add_argument(
        "--gemini_model",
        type=str,
        default=os.environ.get("SCRIBE_MODEL", "gemini-2.0-flash-lite-preview-02-05"),
        help="Gemini model name.",
    )

    openai_group = parser.add_argument_group("OpenAI Provider Options")
    openai_group.add_argument(
        "--openai_api_key",
        help="OpenAI API key (can also set OPENAI_API_KEY env variable).",
        default=os.environ.get("OPENAI_API_KEY"),
    )
    openai_group.add_argument(
        "--openai_model",
        type=str,
        default=os.environ.get("SCRIBE_MODEL", "gpt-4o"),
        help="OpenAI model name.",
    )

    # Legacy argument for backwards compatibility
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("SCRIBE_MODEL", "gemini-2.0-flash-lite-preview-02-05"),
        help=argparse.SUPPRESS,  # Hide from help but keep for backward compatibility
    )

    args = parser.parse_args()

    # Handle API keys for different providers
    if args.provider == "gemini":
        args.gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not args.gemini_api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable or use --gemini_api_key."
            )
    elif args.provider == "openai":
        args.openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not args.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use --openai_api_key."
            )

    # Configure logging
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Main script logic
    if os.path.isfile(args.input_path):
        process_video(args.input_path, args)
    elif os.path.isdir(args.input_path):
        video_files = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv", "*.ts", "*.flv", "*.webm"))
        ]
        for video_file in video_files:
            process_video(video_file, args)
    else:
        logging.error(f"Invalid path to video or directory: {args.input_path}")
