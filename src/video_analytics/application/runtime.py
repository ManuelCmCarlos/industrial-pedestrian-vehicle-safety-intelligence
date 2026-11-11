from dataclasses import dataclass
from time import perf_counter

from video_analytics.application.pipeline import (
    FrameProcessingResult,
    VideoAnalyticsPipeline,
)
from video_analytics.sources.video_source import VideoSource


@dataclass(frozen=True)
class RuntimeFrameResult:
    """
    Result produced for one frame during continuous processing.
    """

    frame_index: int
    source_timestamp: float | None
    processing_seconds: float
    processing_fps: float
    pipeline_result: FrameProcessingResult


class VideoAnalyticsRuntime:
    """
    Run a VideoAnalyticsPipeline continuously on frames produced
    by a VideoSource.
    """

    def __init__(
        self,
        source: VideoSource,
        pipeline: VideoAnalyticsPipeline,
    ):
        self.source = source
        self.pipeline = pipeline

    def run(
        self,
        max_frames: int | None = None,
    ):
        """
        Process frames until the source ends or max_frames is reached.

        Yields:
            RuntimeFrameResult for every processed frame.

        The source is always released when processing finishes,
        including when an exception occurs.
        """

        if max_frames is not None and max_frames <= 0:
            raise ValueError(
                "max_frames must be greater than zero."
            )

        frame_index = 0

        self.source.open()

        source_fps = self.source.fps

        try:
            while True:
                if (
                    max_frames is not None
                    and frame_index >= max_frames
                ):
                    break

                frame = self.source.read()

                if frame is None:
                    break

                if source_fps > 0:
                    source_timestamp = (
                        frame_index / source_fps
                    )
                else:
                    source_timestamp = None

                started_at = perf_counter()

                pipeline_result = self.pipeline.process_frame(
                    frame,
                    timestamp=source_timestamp,
                )

                processing_seconds = (
                    perf_counter() - started_at
                )

                if processing_seconds > 0:
                    processing_fps = (
                        1.0 / processing_seconds
                    )
                else:
                    processing_fps = 0.0

                yield RuntimeFrameResult(
                    frame_index=frame_index,
                    source_timestamp=source_timestamp,
                    processing_seconds=processing_seconds,
                    processing_fps=processing_fps,
                    pipeline_result=pipeline_result,
                )

                frame_index += 1

        finally:
            self.source.release()