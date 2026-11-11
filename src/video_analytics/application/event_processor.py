from dataclasses import dataclass

from video_analytics.events.engine import EventEngine
from video_analytics.events.models import Event
from video_analytics.storage.event_repository import EventRepository
from video_analytics.tracking.models import TrackedObject


@dataclass(frozen=True)
class PersistedEvent:
    """
    Connect one generated Event with its SQLite row ID.
    """

    row_id: int
    event: Event


class EventProcessor:
    """
    Coordinate event generation and event persistence.

    This application-layer component evaluates tracked objects through
    the EventEngine and persists every generated Event using the
    EventRepository.
    """

    def __init__(
        self,
        event_engine: EventEngine,
        event_repository: EventRepository,
    ):
        self.event_engine = event_engine
        self.event_repository = event_repository

    def process(
        self,
        tracked_object: TrackedObject,
    ) -> list[PersistedEvent]:
        """
        Evaluate one tracked object and persist all generated events.

        Returns:
            Generated events together with their database row IDs.
        """

        events = self.event_engine.evaluate(tracked_object)

        persisted_events = []

        for event in events:
            row_id = self.event_repository.save(event)

            persisted_events.append(
                PersistedEvent(
                    row_id=row_id,
                    event=event,
                )
            )

        return persisted_events

    def process_frame(
        self,
        tracked_objects: list[TrackedObject],
        timestamp: float | None = None,
    ) -> list[PersistedEvent]:
        """Persistir eventos que dependem de vários objetos do mesmo frame."""
        events = self.event_engine.evaluate_frame(tracked_objects, timestamp)
        persisted = []
        for event in events:
            row_id = self.event_repository.save(event)
            persisted.append(PersistedEvent(row_id=row_id, event=event))
        return persisted
