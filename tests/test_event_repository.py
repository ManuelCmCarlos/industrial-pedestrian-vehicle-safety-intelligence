from video_analytics.events.models import Event, utc_now
from video_analytics.storage.database import Database
from video_analytics.storage.event_repository import (
    EventRepository,
)


def test_repository_saves_and_reads_event(
    tmp_path,
):
    database_path = tmp_path / "events.db"

    database = Database(
        database_path
    )

    database.initialize()

    repository = EventRepository(
        database
    )

    event = Event(
        timestamp=utc_now(),
        event_type="line_crossing",
        track_id=7,
        object_class="person",
        confidence=0.92,
        source_id="camera_01",
        rule_id="entrance_line",
        direction="down",
        metadata={
            "previous_y": 490,
            "current_y": 515,
            "line_y": 500,
        },
    )

    row_id = repository.save(
        event
    )

    recent_events = repository.get_recent(
        limit=1
    )

    assert row_id == 1
    assert len(recent_events) == 1

    stored_event = recent_events[0]

    assert stored_event.event_type == "line_crossing"
    assert stored_event.track_id == 7
    assert stored_event.object_class == "person"
    assert stored_event.confidence == 0.92
    assert stored_event.source_id == "camera_01"
    assert stored_event.rule_id == "entrance_line"
    assert stored_event.direction == "down"

    assert stored_event.metadata == {
        "previous_y": 490,
        "current_y": 515,
        "line_y": 500,
    }

    assert stored_event.timestamp.tzinfo is not None


def test_repository_returns_newest_events_first(
    tmp_path,
):
    database_path = tmp_path / "events.db"

    database = Database(
        database_path
    )

    database.initialize()

    repository = EventRepository(
        database
    )

    events = [
        Event(
            timestamp=utc_now(),
            event_type="line_crossing",
            track_id=7,
            object_class="person",
            confidence=0.90,
            source_id="camera_01",
            rule_id="entrance_line",
            direction="down",
            metadata={},
        ),
        Event(
            timestamp=utc_now(),
            event_type="line_crossing",
            track_id=8,
            object_class="person",
            confidence=0.88,
            source_id="camera_01",
            rule_id="entrance_line",
            direction="up",
            metadata={},
        ),
        Event(
            timestamp=utc_now(),
            event_type="line_crossing",
            track_id=9,
            object_class="person",
            confidence=0.95,
            source_id="camera_01",
            rule_id="entrance_line",
            direction="down",
            metadata={},
        ),
    ]

    for event in events:
        repository.save(event)

    recent_events = repository.get_recent(
        limit=3
    )

    assert len(recent_events) == 3

    assert [
        event.track_id
        for event in recent_events
    ] == [
        9,
        8,
        7,
    ]