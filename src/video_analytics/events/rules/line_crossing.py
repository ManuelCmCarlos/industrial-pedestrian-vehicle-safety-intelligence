from dataclasses import dataclass

from video_analytics.tracking.models import TrackedObject


@dataclass(frozen=True)
class LineCrossingResult:
    """
    Result produced when a tracked object crosses the configured line.
    """

    track_id: int
    direction: str
    previous_y: int
    current_y: int
    line_y: int


class HorizontalLineCrossingRule:
    """
    Detect crossings of one horizontal line.

    The rule keeps the previous vertical center position for each
    track ID and detects transitions from one side of the line
    to the other.
    """

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.previous_positions: dict[int, int] = {}

    def evaluate(
        self,
        tracked_object: TrackedObject,
    ) -> LineCrossingResult | None:
        """
        Evaluate one tracked object.

        Returns:
            LineCrossingResult when a crossing occurs.
            None when there is no crossing yet.
        """

        track_id = tracked_object.track_id
        current_y = tracked_object.center[1]

        previous_y = self.previous_positions.get(track_id)

        self.previous_positions[track_id] = current_y

        if previous_y is None:
            return None

        if previous_y < self.line_y <= current_y:
            return LineCrossingResult(
                track_id=track_id,
                direction="down",
                previous_y=previous_y,
                current_y=current_y,
                line_y=self.line_y,
            )

        if previous_y > self.line_y >= current_y:
            return LineCrossingResult(
                track_id=track_id,
                direction="up",
                previous_y=previous_y,
                current_y=current_y,
                line_y=self.line_y,
            )

        return None