import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from video_analytics.inference.postprocessing import Detection, CLASS_NAMES
from video_analytics.tracking.models import TrackedObject


TRACK_ACTIVATION_THRESHOLD = 0.60
HIGH_CONF_DET_THRESHOLD = 0.60
MINIMUM_CONSECUTIVE_FRAMES = 2
LOST_TRACK_BUFFER = 30
DEFAULT_FRAME_RATE = 30.0


class ByteTrackAdapter:
    """
    Adapter between Project 4 Detection objects and ByteTrack.

    The rest of Project 4 should interact with this class rather than
    importing ByteTrackTracker or supervision.Detections directly.
    """

    def __init__(
        self,
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
        lost_track_buffer=LOST_TRACK_BUFFER,
        frame_rate=DEFAULT_FRAME_RATE,
        high_conf_det_threshold=HIGH_CONF_DET_THRESHOLD,
    ):
        self.frame_rate = float(frame_rate)

        self.tracker = ByteTrackTracker(
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            lost_track_buffer=lost_track_buffer,
            frame_rate=self.frame_rate,
        )

    def update(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
    ) -> list[TrackedObject]:
        """
        Associate detections with persistent track IDs.

        Args:
            detections:
                Project 4 Detection objects for the current frame.

            timestamp:
                Optional absolute frame time in seconds.

        Returns:
            Project 4 TrackedObject objects with persistent track IDs.
        """

        if not detections:
            supervision_detections = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
        else:
            supervision_detections = sv.Detections(
                xyxy=np.array(
                    [detection.bbox for detection in detections],
                    dtype=np.float32,
                ),
                confidence=np.array(
                    [detection.confidence for detection in detections],
                    dtype=np.float32,
                ),
                class_id=np.array(
                    [detection.class_id for detection in detections],
                    dtype=np.int32,
                ),
            )

        tracked_detections = self.tracker.update(
            supervision_detections,
            timestamp=timestamp,
        )

        tracked_objects = []

        for index in range(len(tracked_detections)):
            track_id = int(
                tracked_detections.tracker_id[index]
            )

            if track_id < 0:
                continue

            x1, y1, x2, y2 = (
                tracked_detections.xyxy[index]
            )

            class_id = int(
                tracked_detections.class_id[index]
            )

            confidence = float(
                tracked_detections.confidence[index]
            )

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = CLASS_NAMES[class_id]

            tracked_object = TrackedObject(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )

            tracked_objects.append(
                tracked_object
            )

        return tracked_objects