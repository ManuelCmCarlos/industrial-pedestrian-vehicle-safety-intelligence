from video_analytics.events.rules.line_crossing import (
    HorizontalLineCrossingRule,
)
from video_analytics.tracking.models import TrackedObject


def make_tracked_object(
    track_id: int,
    center_y: int,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(100, center_y - 100, 200, center_y + 100),
        center=(150, center_y),
        confidence=0.90,
        class_id=0,
        class_name="person",
    )


def test_line_crossing_detects_down_direction():
    rule = HorizontalLineCrossingRule(
        line_y=500
    )

    first = make_tracked_object(
        track_id=7,
        center_y=470,
    )

    second = make_tracked_object(
        track_id=7,
        center_y=490,
    )

    third = make_tracked_object(
        track_id=7,
        center_y=515,
    )

    assert rule.evaluate(first) is None
    assert rule.evaluate(second) is None

    result = rule.evaluate(third)

    assert result is not None
    assert result.track_id == 7
    assert result.direction == "down"
    assert result.previous_y == 490
    assert result.current_y == 515
    assert result.line_y == 500


def test_line_crossing_detects_up_direction():
    rule = HorizontalLineCrossingRule(
        line_y=500
    )

    first = make_tracked_object(
        track_id=9,
        center_y=520,
    )

    second = make_tracked_object(
        track_id=9,
        center_y=505,
    )

    third = make_tracked_object(
        track_id=9,
        center_y=480,
    )

    assert rule.evaluate(first) is None
    assert rule.evaluate(second) is None

    result = rule.evaluate(third)

    assert result is not None
    assert result.track_id == 9
    assert result.direction == "up"
    assert result.previous_y == 505
    assert result.current_y == 480
    assert result.line_y == 500