import numpy as np

from video_analytics.application.pipeline import (
    FrameProcessingResult,
)
from video_analytics.application.runtime import (
    VideoAnalyticsRuntime,
)


class FakeVideoSource:
    def __init__(self):
        self.frames = [
            np.zeros(
                (480, 640, 3),
                dtype=np.uint8,
            )
            for _ in range(5)
        ]

        self.current_index = 0
        self.open_called = False
        self.release_called = False

    @property
    def fps(self):
        return 60.0

    def open(self):
        self.open_called = True

    def read(self):
        if self.current_index >= len(self.frames):
            return None

        frame = self.frames[self.current_index]

        self.current_index += 1

        return frame

    def release(self):
        self.release_called = True


class FakePipeline:
    def __init__(self):
        self.received_frames = []
        self.received_timestamps = []

    def process_frame(
        self,
        frame,
        timestamp=None,
    ):
        self.received_frames.append(frame)

        self.received_timestamps.append(
            timestamp
        )

        return FrameProcessingResult(
            detections=[],
            tracked_objects=[],
            persisted_events=[],
        )


def test_runtime_respects_max_frames_and_releases_source():
    source = FakeVideoSource()

    pipeline = FakePipeline()

    runtime = VideoAnalyticsRuntime(
        source=source,
        pipeline=pipeline,
    )

    results = list(
        runtime.run(
            max_frames=3
        )
    )

    assert source.open_called is True
    assert source.release_called is True

    assert len(results) == 3

    assert [
        result.frame_index
        for result in results
    ] == [
        0,
        1,
        2,
    ]

    assert len(
        pipeline.received_frames
    ) == 3

    assert pipeline.received_timestamps == [
        0.0,
        1.0 / 60.0,
        2.0 / 60.0,
    ]

    assert [
        result.source_timestamp
        for result in results
    ] == [
        0.0,
        1.0 / 60.0,
        2.0 / 60.0,
    ]

    assert all(
        result.processing_seconds >= 0
        for result in results
    )

    assert all(
        result.processing_fps > 0
        for result in results
    )