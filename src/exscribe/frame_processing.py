import logging
from typing import List

import av
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from .models import TimestampedFrame


def get_unique_frames(
    video_path: str,
    similarity_threshold: float = 0.0,
    frame_skip: int = 10,
    scale: float = 1.0,
) -> List[TimestampedFrame]:
    """Extracts unique frames from a video based on structural similarity."""
    logging.info(f"Extracting unique frames from video: {video_path}")
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames

        # Optimize decoding for faster frame access
        video_stream.thread_type = "AUTO"

        unique_frames = []
        previous_frame = None

        # Get time base for accurate timestamp calculation
        time_base = float(video_stream.time_base)
        frame_duration = float(
            video_stream.time_base * video_stream.duration / total_frames
        )

        # Create progress bar
        for frame_index, frame in tqdm(
            enumerate(container.decode(video=0)),
            total=total_frames,
            desc="Extracting frames",
        ):
            if frame_index % frame_skip != 0:  # Skip frames
                continue

            # Calculate timestamp in seconds using PTS (Presentation Time Stamp)
            timestamp = float(frame.pts * time_base)

            # Resize frame
            width = int(frame.width * scale)
            height = int(frame.height * scale)
            frame = frame.reformat(
                width=width, height=height, interpolation=0x1
            )  # fast bilinear

            current_frame = frame.to_ndarray(format="rgb24")

            if previous_frame is None:
                unique_frames.append(
                    TimestampedFrame(
                        frame=current_frame,
                        timestamp=timestamp,
                        frame_number=frame_index,
                    )
                )
                previous_frame = current_frame
                continue

            # Calculate SSIM between current and previous frame
            if similarity_threshold > 0.0:
                # Convert frames to grayscale for SSIM comparison
                gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                similarity = ssim(gray1, gray2)
                if similarity < similarity_threshold:
                    unique_frames.append(
                        TimestampedFrame(
                            frame=current_frame,
                            timestamp=timestamp,
                            frame_number=frame_index,
                        )
                    )
                    previous_frame = current_frame

            else:
                unique_frames.append(
                    TimestampedFrame(
                        frame=current_frame,
                        timestamp=timestamp,
                        frame_number=frame_index,
                    )
                )

        container.close()
        logging.info(f"Extracted {len(unique_frames)} unique frames.")
        return unique_frames

    except Exception as e:
        logging.error(f"Error extracting unique frames: {e}")
        return []


def apply_edge_detection(frames: List[TimestampedFrame]) -> List[TimestampedFrame]:
    """Applies Canny edge detection to a list of frames."""
    processed_frames = []
    for timestamped_frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(timestamped_frame.frame, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detection
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        processed_frames.append(
            TimestampedFrame(
                frame=edges,
                timestamp=timestamped_frame.timestamp,
                frame_number=timestamped_frame.frame_number,
            )
        )
    logging.info("Applied edge detection to frames.")
    return processed_frames
