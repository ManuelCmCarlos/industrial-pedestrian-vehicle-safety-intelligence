from video_analytics.application.event_processor import (
    EventProcessor,
)
from video_analytics.events.engine import EventEngine
from video_analytics.events.rules.line_crossing import (
    HorizontalLineCrossingRule,
)
from video_analytics.storage.database import Database
from video_analytics.storage.event_repository import (
    EventRepository,
)
from video_analytics.tracking.models import TrackedObject


def make_tracked_object(
    track_id: int,
    center_y: int,
    confidence: float,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        bbox=(
            100,
            center_y - 100,
            200,
            center_y + 100,
        ),
        center=(150, center_y),
        confidence=confidence,
        class_id=0,
        class_name="person",
    )


def test_event_processor_persists_generated_event(
    tmp_path,
):
    database = Database(
        tmp_path / "events.db"
    )

    database.initialize()

    repository = EventRepository(
        database
    )

    line_rule = HorizontalLineCrossingRule(
        line_y=500
    )

    event_engine = EventEngine(
        line_rule=line_rule,
        source_id="camera_01",
        line_rule_id="entrance_line",
    )

    processor = EventProcessor(
        event_engine=event_engine,
        event_repository=repository,
    )

    before_crossing = make_tracked_object(
        track_id=10,
        center_y=490,
        confidence=0.91,
    )

    after_crossing = make_tracked_object(
        track_id=10,
        center_y=515,
        confidence=0.93,
    )

    assert processor.process(
        before_crossing
    ) == []

    persisted_events = processor.process(
        after_crossing
    )

    assert len(persisted_events) == 1

    persisted_event = persisted_events[0]

    assert persisted_event.row_id == 1
    assert persisted_event.event.event_type == (
        "line_crossing"
    )
    assert persisted_event.event.track_id == 10
    assert persisted_event.event.direction == "down"

    stored_events = repository.get_recent(
        limit=1
    )

    assert len(stored_events) == 1

    stored_event = stored_events[0]

    assert stored_event.track_id == 10
    assert stored_event.event_type == (
        "line_crossing"
    )
    assert stored_event.direction == "down"
    assert stored_event.confidence == 0.93