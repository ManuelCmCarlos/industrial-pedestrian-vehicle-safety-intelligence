import numpy as np
import pytest

from video_analytics.sources import video_source


class FakeCapture:
    def __init__(self, source):
        self.source = source
        self.opened = True
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.read_count > 0:
            return False, None

        self.read_count += 1

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        return True, frame

    def get(self, property_id):
        if property_id == (
            video_source.cv2.CAP_PROP_FPS
        ):
            return 60.0

        if property_id == (
            video_source.cv2.CAP_PROP_FRAME_COUNT
        ):
            return 100.0

        return 0.0

    def release(self):
        self.released = True
        self.opened = False


def test_video_source_open_read_metadata_and_release(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "test_video.mp4"

    video_path.touch()

    fake_capture = FakeCapture(
        str(video_path)
    )

    monkeypatch.setattr(
        video_source.cv2,
        "VideoCapture",
        lambda source: fake_capture,
    )

    source = video_source.VideoSource(
        str(video_path)
    )

    source.open()

    assert source.capture is fake_capture

    assert source.fps == 60.0
    assert source.frame_count == 100

    frame = source.read()

    assert frame is not None

    assert frame.shape == (
        480,
        640,
        3,
    )

    end_frame = source.read()

    assert end_frame is None

    source.release()

    assert fake_capture.released is True
    assert source.capture is None


def test_video_source_missing_local_file_raises_error(
    tmp_path,
):
    missing_path = (
        tmp_path
        / "missing_video.mp4"
    )

    source = video_source.VideoSource(
        str(missing_path)
    )

    with pytest.raises(
        FileNotFoundError,
        match="Video source file not found",
    ):
        source.open()