from .models import SubtitleEntry, TimestampedFrame
from .frame_processing import get_unique_frames, apply_edge_detection
from .transcription import transcribe_frames
from .subtitle import (
    merge_subtitle_frames,
    post_process,
    convert_to_srt,
)
from .cli import main, process_video

__all__ = [
    "SubtitleEntry",
    "TimestampedFrame",
    "get_unique_frames",
    "apply_edge_detection",
    "transcribe_frames",
    "merge_subtitle_frames",
    "post_process",
    "convert_to_srt",
    "main",
    "process_video",
]
