import json
import logging
import os
from abc import ABC, abstractmethod
from itertools import batched, chain
from time import sleep, time
from typing import Dict, List, Any, Optional, Type

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from PIL import Image
from tqdm import tqdm

from .models import TimestampedFrame
from .providers import TranscriptionProvider, GeminiProvider

# Provider registry
PROVIDERS: Dict[str, Type[TranscriptionProvider]] = {
    GeminiProvider.get_provider_name(): GeminiProvider,
}


def get_provider(provider_name: str, **config) -> TranscriptionProvider:
    """Get a transcription provider instance by name."""
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. Available providers: {', '.join(PROVIDERS.keys())}"
        )

    provider_class = PROVIDERS[provider_name]
    return provider_class(**config)


def transcribe_frames(
    frames: List[TimestampedFrame],
    provider_name: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    language: str = "English",
    retry_delay: int = 5,
    max_retries: int = 3,
    batch_size: int = 10,
    cache_file: str = "transcriptions_cache.json",
    save_every_n: int = 10,
    **provider_config,
) -> List[TimestampedFrame]:
    """Transcribes text from frames using the specified provider with batch processing."""
    # Configure provider
    config = {
        **provider_config,
        "api_key": api_key,
        "model_name": model_name,
        "language": language,
    }

    # Remove None values from config
    config = {k: v for k, v in config.items() if v is not None}

    # Load cached transcriptions if the cache file exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf8") as f:
                cached_transcriptions = json.load(f)
            # Apply cached transcriptions to frames
            for i, transcript in enumerate(cached_transcriptions):
                frames[i].text = transcript["text"]
            logging.info(f"Loaded transcriptions from cache file: {cache_file}")

        except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
            logging.warning(
                f"Failed to load transcriptions from cache file: {e}. Starting from scratch."
            )
            cached_transcriptions = []
    else:
        cached_transcriptions = []

    # Get the appropriate provider
    provider = get_provider(provider_name, **config)

    frame_batches = list(batched(frames, batch_size))
    for batch_index, frame_batch in tqdm(
        enumerate(frame_batches), total=len(frame_batches), desc="Transcribing frames"
    ):
        # Skip already transcribed batches
        if frame_batch[-1].text:
            continue

        retries = 0
        while retries < max_retries:
            try:
                tick = time()
                batch_transcriptions = provider.transcribe_batch(
                    frame_batch, batch_index * batch_size
                )

                # Apply transcriptions to frames in the batch
                for j, frame in enumerate(frame_batch):
                    frame.text = batch_transcriptions[j]["text"]
                cached_transcriptions.extend(batch_transcriptions)
                break  # Exit retry loop on success

            except json.JSONDecodeError as e:
                retries += 1
                logging.error(f"Error decoding JSON response: {e}")

            except Exception as e:
                retries += 1
                time_taken = time() - tick
                logging.warning(
                    f"Error processing frame {batch_index} (attempt {retries}/{max_retries}): {e}"
                )

                if retries < max_retries:
                    wait_duration = max(retry_delay - time_taken, 0)
                    logging.info(f"Waiting {wait_duration:.0f} seconds before retry...")
                    sleep(wait_duration)
                else:
                    logging.error(
                        f"Failed to process frame {batch_index} after {max_retries} attempts"
                    )
                    for frame in frame_batch:
                        frame.text = "Transcription failed."

        # Periodically save transcriptions to cache file
        if batch_index % save_every_n == 0:
            try:
                with open(cache_file, "w", encoding="utf8") as f:
                    json.dump(cached_transcriptions, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logging.error(f"Failed to save transcriptions to cache file: {e}")

    logging.info("Transcription complete.")
    try:
        with open(cache_file, "w", encoding="utf8") as f:
            json.dump(cached_transcriptions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save transcriptions to cache file: {e}")
    return frames
