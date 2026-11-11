from video_analytics.events.rules.zone_intrusion import (
    RectangularZoneIntrusionRule,
)
from video_analytics.tracking.models import TrackedObject


def make_tracked_object(
    track_id: int,
    center_x: int,
    center_y: int,
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
        confidence=0.90,
        class_id=0,
        class_name="person",
    )


def test_zone_intrusion_detects_entry_without_duplicates():
    rule = RectangularZoneIntrusionRule(
        zone_id="restricted_zone",
        x1=400,
        y1=200,
        x2=800,
        y2=600,
    )

    outside = make_tracked_object(
        track_id=12,
        center_x=350,
        center_y=400,
    )

    first_entry = make_tracked_object(
        track_id=12,
        center_x=460,
        center_y=400,
    )

    still_inside = make_tracked_object(
        track_id=12,
        center_x=480,
        center_y=420,
    )

    assert rule.evaluate(outside) is None

    result = rule.evaluate(first_entry)

    assert result is not None
    assert result.track_id == 12
    assert result.zone_id == "restricted_zone"
    assert result.center == (460, 400)

    assert rule.evaluate(still_inside) is None


def test_zone_intrusion_detects_reentry_after_exit():
    rule = RectangularZoneIntrusionRule(
        zone_id="restricted_zone",
        x1=400,
        y1=200,
        x2=800,
        y2=600,
    )

    outside = make_tracked_object(
        track_id=12,
        center_x=350,
        center_y=400,
    )

    first_entry = make_tracked_object(
        track_id=12,
        center_x=460,
        center_y=400,
    )

    exited = make_tracked_object(
        track_id=12,
        center_x=870,
        center_y=420,
    )

    second_entry = make_tracked_object(
        track_id=12,
        center_x=650,
        center_y=420,
    )

    assert rule.evaluate(outside) is None
    assert rule.evaluate(first_entry) is not None
    assert rule.evaluate(exited) is None

    result = rule.evaluate(second_entry)

    assert result is not None
    assert result.track_id == 12
    assert result.zone_id == "restricted_zone"
    assert result.center == (650, 420)