from dataclasses import dataclass

import numpy as np

from video_analytics.application.event_processor import (
    EventProcessor,
    PersistedEvent,
)
from video_analytics.inference.detector import ONNXDetector
from video_analytics.inference.postprocessing import Detection
from video_analytics.tracking.models import TrackedObject
from video_analytics.tracking.tracker import ByteTrackAdapter


@dataclass(frozen=True)
class FrameProcessingResult:
    """
    Result produced after processing one video frame.

    It exposes the main runtime information needed by higher-level
    application components such as visualization or a dashboard.
    """

    detections: list[Detection]
    tracked_objects: list[TrackedObject]
    persisted_events: list[PersistedEvent]


class VideoAnalyticsPipeline:
    """
    Coordinate detection, tracking, event processing, and persistence
    for individual video frames.
    """

    def __init__(
        self,
        detector: ONNXDetector,
        tracker: ByteTrackAdapter,
        event_processor: EventProcessor,
    ):
        self.detector = detector
        self.tracker = tracker
        self.event_processor = event_processor

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float | None = None,
    ) -> FrameProcessingResult:
        """
        Process one frame through the complete analytics pipeline.

        Args:
            frame:
                OpenCV BGR image.

            timestamp:
                Optional absolute frame time in seconds for tracking.

        Returns:
            FrameProcessingResult containing detections, tracked
            objects, and any events persisted during this frame.
        """

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a NumPy array."
            )

        detections = self.detector.detect(frame)

        tracked_objects = self.tracker.update(
            detections,
            timestamp=timestamp,
        )

        persisted_events = []

        for tracked_object in tracked_objects:
            persisted_events.extend(
                self.event_processor.process(tracked_object)
            )

        # Regras de interacao recebem a lista completa uma vez por frame.
        persisted_events.extend(
            self.event_processor.process_frame(tracked_objects, timestamp=timestamp)
        )

        return FrameProcessingResult(
            detections=detections,
            tracked_objects=tracked_objects,
            persisted_events=persisted_events,
        )