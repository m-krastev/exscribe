import json
from itertools import chain
from typing import Dict, List, Any

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from PIL import Image

from .base import TranscriptionProvider
from ..models import TimestampedFrame


class GeminiProvider(TranscriptionProvider):
    """Gemini API provider for transcription."""

    def __init__(self, api_key: str, model_name: str, language: str, **kwargs):
        """Initialize the Gemini provider.

        Args:
            api_key: Gemini API key
            model_name: Name of the Gemini model to use
            language: Language of the subtitles to transcribe
            **kwargs: Additional configuration options
        """
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                enum=[],
                required=["subtitles"],
                properties={
                    "subtitles": content.Schema(
                        type=content.Type.ARRAY,
                        items=content.Schema(
                            type=content.Type.OBJECT,
                            enum=[],
                            required=["image", "text"],
                            properties={
                                "image": content.Schema(type=content.Type.NUMBER),
                                "text": content.Schema(type=content.Type.STRING),
                            },
                        ),
                    ),
                },
            ),
            "response_mime_type": "application/json",
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=f"Transcribe the subtitle text from language {language}. Respond for each image. Preserve formatting such as italics or slanted text with the <i> tags. Preserve new lines. If the text repeats, respond with it once and then use 1 for every consecutive same subtitle. If there is no subtitle text, consistently respond with 0.",
        )

    def transcribe_batch(
        self, frames: List[TimestampedFrame], batch_index: int
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of frames using Gemini API."""
        input_data = list(
            chain.from_iterable(
                (f"image: {j}", Image.fromarray(frame.frame))
                for j, frame in enumerate(frames, start=batch_index)
            )
        )

        chat_session = self.model.start_chat()
        response = chat_session.send_message(input_data)
        transcriptions = json.loads(response.text.strip())["subtitles"]

        return transcriptions

    @classmethod
    def get_provider_name(cls) -> str:
        return "gemini"
