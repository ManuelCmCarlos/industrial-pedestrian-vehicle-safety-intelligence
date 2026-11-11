"""Testes mínimos da regra preventiva, sem modelo ONNX."""

from video_analytics.events.engine import EventEngine
from video_analytics.events.rules.shared_aisle_warning import SharedAisleWarningRule
from video_analytics.tracking.models import TrackedObject

POLYGON = [[890, 300], [1150, 300], [1320, 840], [760, 840]]


def track(identifier: int, klass: str, foot_x: int, foot_y: int) -> TrackedObject:
    bbox = (foot_x - 20, foot_y - 60, foot_x + 20, foot_y)
    return TrackedObject(
        track_id=identifier,
        bbox=bbox,
        center=(foot_x, foot_y - 30),
        confidence=0.9,
        class_id=0 if klass == "person" else 1,
        class_name=klass,
    )


def new_rule(rearm_after_frames: int = 20) -> SharedAisleWarningRule:
    return SharedAisleWarningRule(
        polygon_pixels=POLYGON,
        confirmation_frames=3,
        motion_window_frames=4,
        min_vehicle_displacement_px=12,
        rearm_after_frames=rearm_after_frames,
    )


def pair(vehicle_x: int = 1140):
    return [track(0, "person", 931, 500), track(1, "industrial_vehicle", vehicle_x, 658)]


def test_three_frames_and_movement_emit_one_event():
    rule = new_rule()
    engine = EventEngine(line_rule=None, source_id="test", shared_aisle_rule=rule)
    positions = [1140, 1141, 1142, 1155, 1160]
    events_by_frame = [engine.evaluate_frame(pair(x), i / 10.0) for i, x in enumerate(positions)]
    assert [len(events) for events in events_by_frame] == [0, 0, 0, 1, 0]
    event = events_by_frame[3][0]
    assert event.event_type == "pedestrian_vehicle_shared_aisle_warning"
    assert event.metadata["frame_index"] == 3
    assert event.metadata["source_time_seconds"] == 0.3
    assert event.metadata["person_track_id"] == 0
    assert event.metadata["vehicle_track_id"] == 1
    assert event.metadata["vehicle_image_displacement_px"] == 15.0


def test_stationary_vehicle_does_not_alert():
    rule = new_rule()
    assert all(rule.evaluate(pair(), i / 10.0) is None for i in range(10))


def test_missing_track_interrupts_confirmation():
    rule = new_rule()
    assert rule.evaluate(pair(1140), 0.0) is None
    assert rule.evaluate([], 0.1) is None
    assert rule.evaluate(pair(1140), 0.2) is None
    assert rule.evaluate(pair(1141), 0.3) is None
    assert rule.evaluate(pair(1155), 0.4) is not None


def test_no_alert_without_both_classes_or_outside_polygon():
    rule = new_rule()
    for i in range(6):
        assert rule.evaluate([track(0, "person", 931, 500)], i / 10.0) is None
    for i in range(6, 12):
        assert rule.evaluate([
            track(0, "person", 100, 500),
            track(1, "industrial_vehicle", 1140 + 20 * i, 658),
        ], i / 10.0) is None


def test_rearms_only_after_sustained_absence():
    rule = new_rule(rearm_after_frames=4)
    for i, x in enumerate([1140, 1141, 1142]):
        assert rule.evaluate(pair(x), i / 10.0) is None
    assert rule.evaluate(pair(1155), 0.3) is not None
    assert rule.evaluate([], 0.4) is None
    assert rule.evaluate([], 0.5) is None
    assert rule.evaluate(pair(1140), 0.6) is None
    assert rule.evaluate(pair(1160), 0.7) is None
    for i in range(4):
        assert rule.evaluate([], 0.8 + i / 10.0) is None
    for i, x in enumerate([1140, 1141, 1142]):
        assert rule.evaluate(pair(x), 1.2 + i / 10.0) is None
    assert rule.evaluate(pair(1155), 1.5) is not None


def test_original_event_engine_without_rule_is_compatible():
    engine = EventEngine(line_rule=None, source_id="test")
    assert engine.evaluate_frame(pair(1140), 0.0) == []
    assert engine.evaluate(track(0, "person", 931, 500)) == []


def test_event_processor_persists_existing_and_pair_events(tmp_path):
    """EventProcessor com SQLite real; não requer modelo ONNX."""
    from video_analytics.application.event_processor import EventProcessor
    from video_analytics.events.rules.zone_intrusion import RectangularZoneIntrusionRule
    from video_analytics.storage.database import Database
    from video_analytics.storage.event_repository import EventRepository

    db = Database(tmp_path / "events.db")
    db.initialize()
    repo = EventRepository(db)
    engine = EventEngine(
        line_rule=None,
        source_id="test",
        zone_rule=RectangularZoneIntrusionRule("existing_zone", 0, 0, 1900, 900),
        shared_aisle_rule=new_rule(),
    )
    processor = EventProcessor(engine, repo)

    output_counts = []
    for index, x in enumerate([1140, 1141, 1142, 1155]):
        tracks = pair(x)
        persisted = []
        for item in tracks:
            persisted.extend(processor.process(item))
        persisted.extend(processor.process_frame(tracks, timestamp=index / 10.0))
        output_counts.append(len(persisted))

    assert output_counts == [2, 0, 0, 1]
    saved = repo.get_recent(limit=10)
    assert len(saved) == 3
    assert [event.event_type for event in saved].count("zone_intrusion") == 2
    assert [event.event_type for event in saved].count(
        "pedestrian_vehicle_shared_aisle_warning"
    ) == 1
