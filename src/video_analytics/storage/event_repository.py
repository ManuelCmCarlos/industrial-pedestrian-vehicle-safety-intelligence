import json
from dataclasses import dataclass
from datetime import datetime

from video_analytics.events.models import Event
from video_analytics.storage.database import Database


@dataclass(frozen=True)
class StoredEvent:
    """Represent an Event retrieved from SQLite with its database row ID."""

    row_id: int
    event: Event


class EventRepository:
    """
    Persist and retrieve Project 4 Event objects using SQLite.
    """

    def __init__(self, database: Database):
        self.database = database

    def save(self, event: Event) -> int:
        """
        Save one Event and return the generated database row ID.
        """

        metadata_json = None

        if event.metadata is not None:
            metadata_json = json.dumps(event.metadata)

        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    timestamp,
                    event_type,
                    track_id,
                    object_class,
                    confidence,
                    source_id,
                    rule_id,
                    direction,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.track_id,
                    event.object_class,
                    event.confidence,
                    event.source_id,
                    event.rule_id,
                    event.direction,
                    metadata_json,
                ),
            )

            connection.commit()

            return int(cursor.lastrowid)

    def get_recent(self, limit: int = 20) -> list[Event]:
        """
        Return the most recently stored Event objects, newest first.

        Keep the existing return type for callers that need only Events.
        """

        return [record.event for record in self.get_recent_records(limit)]

    def get_recent_records(self, limit: int = 20) -> list[StoredEvent]:
        """
        Return stored events with their SQLite row IDs, newest first.
        """

        if limit <= 0:
            return []

        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    timestamp,
                    event_type,
                    track_id,
                    object_class,
                    confidence,
                    source_id,
                    rule_id,
                    direction,
                    metadata
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records = []

        for row in rows:
            (
                row_id,
                timestamp,
                event_type,
                track_id,
                object_class,
                confidence,
                source_id,
                rule_id,
                direction,
                metadata_json,
            ) = row

            metadata = None

            if metadata_json is not None:
                metadata = json.loads(metadata_json)

            event = Event(
                timestamp=datetime.fromisoformat(timestamp),
                event_type=event_type,
                track_id=track_id,
                object_class=object_class,
                confidence=confidence,
                source_id=source_id,
                rule_id=rule_id,
                direction=direction,
                metadata=metadata,
            )

            records.append(StoredEvent(row_id=int(row_id), event=event))

        return records
