import numpy as np

from video_analytics.application.pipeline import (
    VideoAnalyticsPipeline,
)
from video_analytics.inference.postprocessing import Detection
from video_analytics.tracking.models import TrackedObject


class FakeDetector:
    def __init__(self):
        self.received_frame = None

    def detect(self, frame):
        self.received_frame = frame

        return [
            Detection(
                bbox=(100, 100, 200, 300),
                confidence=0.90,
                class_id=0,
                class_name="person",
            )
        ]


class FakeTracker:
    def __init__(self):
        self.received_detections = None
        self.received_timestamp = None

    def update(
        self,
        detections,
        timestamp=None,
    ):
        self.received_detections = detections
        self.received_timestamp = timestamp

        return [
            TrackedObject(
                track_id=7,
                bbox=(100, 100, 200, 300),
                center=(150, 200),
                confidence=0.90,
                class_id=0,
                class_name="person",
            )
        ]


class FakeEventProcessor:
    def __init__(self):
        self.received_tracks = []

    def process(self, tracked_object):
        self.received_tracks.append(
            tracked_object
        )
        return []

    def process_frame(self, tracked_objects, timestamp=None):
        return []


def test_pipeline_coordinates_processing_components():
    detector = FakeDetector()
    tracker = FakeTracker()
    event_processor = FakeEventProcessor()

    pipeline = VideoAnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        event_processor=event_processor,
    )

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    result = pipeline.process_frame(
        frame,
        timestamp=1.25,
    )

    assert detector.received_frame is frame

    assert len(
        tracker.received_detections
    ) == 1

    assert (
        tracker.received_detections[0].class_name
        == "person"
    )

    assert tracker.received_timestamp == 1.25

    assert len(
        event_processor.received_tracks
    ) == 1

    assert (
        event_processor.received_tracks[0].track_id
        == 7
    )

    assert len(result.detections) == 1
    assert len(result.tracked_objects) == 1
    assert len(result.persisted_events) == 0

    assert result.detections[0].class_name == (
        "person"
    )

    assert result.tracked_objects[0].track_id == 7