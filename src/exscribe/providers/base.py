from abc import ABC, abstractmethod
from typing import Dict, List, Any

from ..models import TimestampedFrame


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the provider with appropriate configuration."""
        pass

    @abstractmethod
    def transcribe_batch(
        self, frames: List[TimestampedFrame], batch_index: int
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of frames and return transcription results.

        Args:
            frames: List of frames to transcribe
            batch_index: Starting index of the batch in the overall frame list

        Returns:
            List of dictionaries with at least a 'text' key for each transcribed frame
        """
        pass

    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """Return the name of the provider."""
        pass
