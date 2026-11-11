from video_analytics.events.models import Event, utc_now
from video_analytics.events.rules.line_crossing import (
    HorizontalLineCrossingRule,
)
from video_analytics.events.rules.zone_intrusion import (
    RectangularZoneIntrusionRule,
)
from video_analytics.events.rules.shared_aisle_warning import (
    SharedAisleWarningRule,
)
from video_analytics.tracking.models import TrackedObject


class EventEngine:
    """
    Coordinate monitoring rules and convert rule results into
    application-level Event objects.
    """

    def __init__(
        self,
        line_rule: HorizontalLineCrossingRule | None,
        source_id: str,
        line_rule_id: str | None = None,
        zone_rule: RectangularZoneIntrusionRule | None = None,
        shared_aisle_rule: SharedAisleWarningRule | None = None,
        shared_aisle_rule_id: str = "shared_aisle_warning",
    ):
        self.line_rule = line_rule
        self.source_id = source_id
        self.line_rule_id = line_rule_id
        self.zone_rule = zone_rule
        self.shared_aisle_rule = shared_aisle_rule
        self.shared_aisle_rule_id = shared_aisle_rule_id

    def evaluate(
        self,
        tracked_object: TrackedObject,
    ) -> list[Event]:
        """
        Evaluate one tracked object against configured monitoring rules.

        Returns:
            A list of generated Event objects.
        """

        events = []

        if self.line_rule is not None:
            line_result = self.line_rule.evaluate(tracked_object)

            if line_result is not None:
                event = Event(
                    timestamp=utc_now(),
                    event_type="line_crossing",
                    track_id=tracked_object.track_id,
                    object_class=tracked_object.class_name,
                    confidence=tracked_object.confidence,
                    source_id=self.source_id,
                    rule_id=self.line_rule_id or "line_rule",
                    direction=line_result.direction,
                    metadata={
                        "previous_y": line_result.previous_y,
                        "current_y": line_result.current_y,
                        "line_y": line_result.line_y,
                    },
                )

                events.append(event)

        if self.zone_rule is not None:
            zone_result = self.zone_rule.evaluate(tracked_object)

            if zone_result is not None:
                event = Event(
                    timestamp=utc_now(),
                    event_type="zone_intrusion",
                    track_id=tracked_object.track_id,
                    object_class=tracked_object.class_name,
                    confidence=tracked_object.confidence,
                    source_id=self.source_id,
                    rule_id=zone_result.zone_id,
                    direction=None,
                    metadata={
                        "center_x": zone_result.center[0],
                        "center_y": zone_result.center[1],
                    },
                )

                events.append(event)

        return events

    def evaluate_frame(
        self,
        tracked_objects: list[TrackedObject],
        timestamp: float | None = None,
    ) -> list[Event]:
        """Avaliar uma interação entre objetos do mesmo frame.

        Este caminho é adicional: evaluate() mantém as regras individuais.
        """
        if self.shared_aisle_rule is None:
            return []

        result = self.shared_aisle_rule.evaluate(tracked_objects, timestamp)
        if result is None:
            return []

        event = Event(
            timestamp=utc_now(),
            event_type="pedestrian_vehicle_shared_aisle_warning",
            track_id=result.person_track_id,
            object_class="person",
            confidence=result.confidence,
            source_id=self.source_id,
            rule_id=self.shared_aisle_rule_id,
            metadata={
                "frame_index": result.frame_index,
                "source_time_seconds": result.source_time_seconds,
                "person_track_id": result.person_track_id,
                "vehicle_track_id": result.vehicle_track_id,
                "person_bottom_center": list(result.person_bottom_center),
                "vehicle_bottom_center": list(result.vehicle_bottom_center),
                "consecutive_frames": result.consecutive_frames,
                "vehicle_image_displacement_px": result.vehicle_image_displacement_px,
                "displacement_threshold_px": (
                    self.shared_aisle_rule.min_vehicle_displacement_px
                ),
                "polygon_pixels": (
                    self.shared_aisle_rule.polygon_pixels.tolist()
                ),
                "meaning": (
                    "circulacao partilhada com deslocamento visual; "
                    "nao confirma proximidade fisica nem near-miss"
                ),
            },
        )
        return [event]
