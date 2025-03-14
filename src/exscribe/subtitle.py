from collections import defaultdict
from typing import List

from .models import SubtitleEntry, TimestampedFrame


def merge_subtitle_frames(frames: List[TimestampedFrame]) -> List[SubtitleEntry]:
    """Merges frames with identical subtitle text and create timestamp ranges."""
    # Group frames by subtitle text
    text_groups = defaultdict(list)
    previous_text = ""
    previous_index = 0

    for frame in frames:
        normalized_text = frame.text.strip().replace("\n", "").replace(" ", "")
        if (
            not normalized_text or normalized_text == "0"
        ):  # Skip empty or whitespace-only text or '0'
            continue

        if frame.text == "1" or frame.text == previous_text:
            frame.text = previous_text  # Use previous text for "1" or repeating text
        else:
            previous_text = frame.text
            previous_index = frame.frame_number

        text_groups[(previous_index, previous_text.strip())].append(frame)

    merged_subtitles = []
    for (_, text), frame_group in text_groups.items():
        # Sort frames by timestamp
        frame_group.sort(key=lambda x: x.timestamp)

        # Start and end timestamps
        start_time = frame_group[0].timestamp
        end_time = frame_group[-1].timestamp

        merged_subtitles.append(
            SubtitleEntry(
                text=text,
                start_time=start_time,
                end_time=end_time,
                frames=[frame.frame_number for frame in frame_group],
            )
        )

    # Sort by start time
    merged_subtitles.sort(key=lambda x: x.start_time)
    return merged_subtitles


def post_process(
    subtitles: List[SubtitleEntry], extend_prev: float = 0.2, extend_next: float = 0.2
) -> List[SubtitleEntry]:
    """Post-processes subtitle entries by extending timestamps and merging consecutively similar entries, including checks for semantic overlap."""
    processed_subtitles: list[SubtitleEntry] = []
    previous_entry = None

    for entry in subtitles:
        if previous_entry and (
            (entry.text == previous_entry.text)
            or len(set(entry.text.split()) & set(previous_entry.text.split())) > 2
        ):
            # Extend previous entry end time
            previous_entry.end_time = entry.end_time
            previous_entry.frames.extend(entry.frames)
        else:
            processed_subtitles.append(entry)
            previous_entry = entry

    for i, entry in enumerate(processed_subtitles):
        # Extend timestamps
        dist_prev = (
            (entry.start_time - processed_subtitles[i - 1].end_time)
            if i > 0
            else extend_prev
        )
        dist_next = (
            (processed_subtitles[i + 1].start_time - entry.end_time)
            if i < len(processed_subtitles) - 1
            else extend_next
        )
        entry.start_time -= min(dist_prev, extend_prev)
        entry.end_time += min(dist_next, extend_next)

    return processed_subtitles


def convert_to_srt(subtitles: List[SubtitleEntry]) -> str:
    """Converts a list of subtitle entries to SRT format."""
    return "\n\n".join([
        entry.to_srt_string(index) for index, entry in enumerate(subtitles, start=1)
    ])
