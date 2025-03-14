from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float
    frame_number: int
    text: str = ""


@dataclass
class SubtitleEntry:
    text: str
    start_time: float
    end_time: float
    frames: List[int] = field(default_factory=list)

    def to_srt_string(self, index: int) -> str:
        """Converts a subtitle entry to SRT format."""

        def convert_to_srt_timestamp(timestamp: float) -> str:
            """Convert timestamp in seconds to SRT format (HH:MM:SS,mmm)"""
            hours, remainder = divmod(timestamp, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02.0f}:{minutes:02.0f}:{seconds:06.3f}".replace(",", ".")

        start_time_str = convert_to_srt_timestamp(self.start_time)
        end_time_str = convert_to_srt_timestamp(self.end_time)

        return f"{index}\n{start_time_str} --> {end_time_str}\n{self.text}"

    def to_json(self) -> dict:
        """Converts a subtitle entry to JSON format."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frames": self.frames,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SubtitleEntry":
        """Creates a subtitle entry from JSON data."""
        return cls(
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            frames=data["frames"],
        )
