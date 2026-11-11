from dataclasses import dataclass

from video_analytics.tracking.models import TrackedObject


@dataclass(frozen=True)
class ZoneIntrusionResult:
    """
    Result produced when a tracked object enters a configured zone.
    """

    track_id: int
    zone_id: str
    center: tuple[int, int]


class RectangularZoneIntrusionRule:
    """
    Detect entry into one rectangular monitoring zone.

    The rule generates one intrusion result when a tracked object
    transitions from outside the zone to inside the zone.
    """

    def __init__(
        self,
        zone_id: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ):
        self.zone_id = zone_id
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

        self.inside_tracks: set[int] = set()

    def evaluate(
        self,
        tracked_object: TrackedObject,
    ) -> ZoneIntrusionResult | None:
        """
        Evaluate one tracked object's center against the zone.

        Returns:
            ZoneIntrusionResult when the object enters the zone.
            None otherwise.
        """

        track_id = tracked_object.track_id
        center_x, center_y = tracked_object.center

        is_inside = (
            self.x1 <= center_x <= self.x2
            and
            self.y1 <= center_y <= self.y2
        )

        was_inside = track_id in self.inside_tracks

        if is_inside and not was_inside:
            self.inside_tracks.add(track_id)

            return ZoneIntrusionResult(
                track_id=track_id,
                zone_id=self.zone_id,
                center=tracked_object.center,
            )

        if not is_inside and was_inside:
            self.inside_tracks.remove(track_id)

        return None