"""Video Loader and Frame Streamer Module.

Provides memory-safe, generator-based video ingestion for arbitrary frame intervals
using zero-based, end-exclusive indexing: [start_frame, end_frame).
Supports high-performance downscaling (e.g. 4K to 1080p) while tracking scale factors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class FramePacket:
    """Encapsulates a decoded frame and its dual-timeline metadata."""
    frame: np.ndarray                      # Decoded image (BGR) at inference resolution
    global_frame_idx: int                  # Zero-based index in the original source video
    segment_frame_idx: int                 # Zero-based index relative to segment start
    global_timestamp_sec: float            # Timestamp in source video (seconds)
    segment_timestamp_sec: float           # Timestamp in segment (seconds)
    original_dims: Tuple[int, int]         # (width, height) of source video
    scaled_dims: Tuple[int, int]           # (width, height) of returned frame
    scale_factors: Tuple[float, float]     # (scale_x, scale_y) to map back to original


def rescale_bbox(
    bbox: Union[List[float], Tuple[float, ...], np.ndarray],
    scale_x: float,
    scale_y: float
) -> List[float]:
    """Rescales a bounding box [x1, y1, x2, y2] from inference to original resolution."""
    x1, y1, x2, y2 = bbox[:4]
    return [
        float(x1 * scale_x),
        float(y1 * scale_y),
        float(x2 * scale_x),
        float(y2 * scale_y),
    ]


def normalize_bbox(
    bbox: Union[List[float], Tuple[float, ...], np.ndarray],
    width: float,
    height: float
) -> List[float]:
    """Converts pixel bounding box [x1, y1, x2, y2] to normalized coordinates in [0.0, 1.0]."""
    x1, y1, x2, y2 = bbox[:4]
    return [
        round(float(np.clip(x1 / width, 0.0, 1.0)), 4),
        round(float(np.clip(y1 / height, 0.0, 1.0)), 4),
        round(float(np.clip(x2 / width, 0.0, 1.0)), 4),
        round(float(np.clip(y2 / height, 0.0, 1.0)), 4),
    ]


class VideoSegmentLoader:
    """Memory-safe video frame streamer with exact interval slicing and resolution scaling."""

    def __init__(
        self,
        video_path: Union[str, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the video segment loader.

        Args:
            video_path: Path to the input video file.
            start_frame: Inclusive start frame index (0-based).
            end_frame: Exclusive end frame index (0-based). If None, reads to video end.
            target_resolution: Optional (width, height) to resize frames for inference.
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path.resolve()}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        # Extract native properties
        self.native_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate intervals: [start_frame, end_frame)
        self.start_frame = int(start_frame)
        if self.start_frame < 0 or self.start_frame >= self.total_frames:
            raise ValueError(
                f"Invalid start_frame {self.start_frame}. Video has {self.total_frames} frames."
            )

        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(int(end_frame), self.total_frames)

        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be strictly less than end_frame ({self.end_frame})."
            )

        self.expected_frames = self.end_frame - self.start_frame
        self.target_resolution = target_resolution

        # Compute scaling factors
        if self.target_resolution is not None:
            target_w, target_h = self.target_resolution
            self.scale_x = self.native_width / float(target_w)
            self.scale_y = self.native_height / float(target_h)
            self.scaled_dims = (target_w, target_h)
        else:
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.scaled_dims = (self.native_width, self.native_height)

    @property
    def duration_sec(self) -> float:
        """Calculates expected segment duration in seconds."""
        return self.expected_frames / self.native_fps if self.native_fps > 0 else 0.0

    def stream_frames(self) -> Generator[FramePacket, None, None]:
        """
        Streams frames from start_frame up to end_frame (exclusive).
        Yields FramePacket dataclass instances one frame at a time to prevent memory leaks.
        """
        # Seek directly to start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        current_global_idx = self.start_frame
        while current_global_idx < self.end_frame:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Video ended prematurely
                break

            segment_idx = current_global_idx - self.start_frame
            global_time = current_global_idx / self.native_fps
            segment_time = segment_idx / self.native_fps

            # Apply resizing if target resolution is configured
            if self.target_resolution is not None and self.scaled_dims != (self.native_width, self.native_height):
                frame_out = cv2.resize(
                    frame,
                    self.scaled_dims,
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                frame_out = frame

            packet = FramePacket(
                frame=frame_out,
                global_frame_idx=current_global_idx,
                segment_frame_idx=segment_idx,
                global_timestamp_sec=round(global_time, 4),
                segment_timestamp_sec=round(segment_time, 4),
                original_dims=(self.native_width, self.native_height),
                scaled_dims=self.scaled_dims,
                scale_factors=(self.scale_x, self.scale_y)
            )

            yield packet
            current_global_idx += 1

    def close(self) -> None:
        """Releases the underlying OpenCV capture object."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"VideoSegmentLoader(video='{self.video_path.name}', "
            f"range=[{self.start_frame}, {self.end_frame}), "
            f"frames={self.expected_frames}, "
            f"fps={self.native_fps:.2f}, "
            f"duration={self.duration_sec:.2f}s, "
            f"native_res={self.native_width}x{self.native_height}, "
            f"scaled_res={self.scaled_dims[0]}x{self.scaled_dims[1]})"
        )
