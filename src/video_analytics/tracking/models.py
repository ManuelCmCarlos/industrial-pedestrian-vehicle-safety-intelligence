from dataclasses import dataclass


@dataclass(frozen=True)
class TrackedObject:
    """
    One object that has been associated with a persistent track ID.

    bbox uses absolute image coordinates:
    (x1, y1, x2, y2)

    center represents the center point of the bounding box:
    (center_x, center_y)
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    class_id: int
    class_name: str