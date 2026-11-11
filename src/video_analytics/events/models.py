from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Event:
    """
    Application-level event produced by the monitoring system.

    This object is intentionally generic so that different rule types
    can be represented in a common format for storage, display, and
    later querying.
    """

    timestamp: datetime
    event_type: str
    track_id: int
    object_class: str
    confidence: float
    source_id: str
    rule_id: str
    direction: str | None = None
    metadata: dict[str, Any] | None = None


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)