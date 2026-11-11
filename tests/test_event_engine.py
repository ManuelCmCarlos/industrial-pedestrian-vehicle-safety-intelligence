from video_analytics.events.engine import EventEngine
from video_analytics.events.rules.line_crossing import (
    HorizontalLineCrossingRule,
)
from video_analytics.events.rules.zone_intrusion import (
    RectangularZoneIntrusionRule,
)
from video_analytics.tracking.models import TrackedObject


def make_tracked_object(
    track_id: int,
    center_x: int,
    center_y: int,
    confidence: float = 0.90,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(
            center_x - 50,
            center_y - 100,
            center_x + 50,
            center_y + 100,
        ),
        center=(center_x, center_y),
        confidence=confidence,
        class_id=0,
        class_name="person",
    )


def test_event_engine_creates_line_crossing_event():
    line_rule = HorizontalLineCrossingRule(
        line_y=500
    )

    engine = EventEngine(
        line_rule=line_rule,
        source_id="camera_01",
        line_rule_id="entrance_line",
    )

    before_crossing = make_tracked_object(
        track_id=7,
        center_x=150,
        center_y=490,
        confidence=0.91,
    )

    after_crossing = make_tracked_object(
        track_id=7,
        center_x=150,
        center_y=515,
        confidence=0.92,
    )

    assert engine.evaluate(before_crossing) == []

    events = engine.evaluate(after_crossing)

    assert len(events) == 1

    event = events[0]

    assert event.event_type == "line_crossing"
    assert event.track_id == 7
    assert event.object_class == "person"
    assert event.confidence == 0.92
    assert event.source_id == "camera_01"
    assert event.rule_id == "entrance_line"
    assert event.direction == "down"

    assert event.metadata == {
        "previous_y": 490,
        "current_y": 515,
        "line_y": 500,
    }

    assert event.timestamp.tzinfo is not None


def test_event_engine_can_create_multiple_events():
    line_rule = HorizontalLineCrossingRule(
        line_y=500
    )

    zone_rule = RectangularZoneIntrusionRule(
        zone_id="restricted_zone",
        x1=400,
        y1=500,
        x2=800,
        y2=700,
    )

    engine = EventEngine(
        line_rule=line_rule,
        source_id="camera_01",
        line_rule_id="entrance_line",
        zone_rule=zone_rule,
    )

    before = make_tracked_object(
        track_id=20,
        center_x=350,
        center_y=480,
        confidence=0.90,
    )

    after = make_tracked_object(
        track_id=20,
        center_x=460,
        center_y=520,
        confidence=0.93,
    )

    assert engine.evaluate(before) == []

    events = engine.evaluate(after)

    assert len(events) == 2

    assert [
        event.event_type
        for event in events
    ] == [
        "line_crossing",
        "zone_intrusion",
    ]

    assert [
        event.rule_id
        for event in events
    ] == [
        "entrance_line",
        "restricted_zone",
    ]

    assert all(
        event.track_id == 20
        for event in events
    )

    assert all(
        event.source_id == "camera_01"
        for event in events
    )