from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """
    Open and read frames from an OpenCV-compatible video source.

    Supported source forms include:
        - video file paths
        - webcam/device indices
        - RTSP/IP-camera URLs supported by OpenCV

    This class handles frame acquisition only.
    It does not perform inference, tracking, event processing,
    visualization, or storage.
    """

    def __init__(self, source):
        self.source = source
        self.capture = None

    def open(self) -> None:
        """
        Open the configured video source.

        Raises:
            FileNotFoundError:
                When a local file path does not exist.

            RuntimeError:
                When OpenCV cannot open the source.
        """

        if isinstance(self.source, str):
            source_path = Path(self.source)

            is_probable_stream = self.source.startswith(
                (
                    "rtsp://",
                    "http://",
                    "https://",
                )
            )

            if not is_probable_stream and not source_path.is_file():
                raise FileNotFoundError(
                    f"Video source file not found: {self.source}"
                )

        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None

            raise RuntimeError(
                f"Unable to open video source: {self.source}"
            )

    def read(self) -> np.ndarray | None:
        """
        Read the next frame.

        Returns:
            OpenCV BGR frame when available.
            None when the stream or video has ended.

        Raises:
            RuntimeError:
                When read() is called before open().
        """

        if self.capture is None:
            raise RuntimeError(
                "Video source is not open."
            )

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        """
        Release the underlying OpenCV capture resource.
        """

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def fps(self) -> float:
        """
        Return the source FPS reported by OpenCV.

        Returns 0.0 when the source is not open or FPS is unavailable.
        """

        if self.capture is None:
            return 0.0

        fps = self.capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            return 0.0

        return float(fps)

    @property
    def frame_count(self) -> int | None:
        """
        Return total frame count when available.

        Live sources may not provide a meaningful frame count.
        """

        if self.capture is None:
            return None

        count = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if count <= 0:
            return None

        return count