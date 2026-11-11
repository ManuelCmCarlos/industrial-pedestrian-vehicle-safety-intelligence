from dataclasses import dataclass

from video_analytics.application.event_processor import EventProcessor
from video_analytics.application.pipeline import VideoAnalyticsPipeline
from video_analytics.application.runtime import VideoAnalyticsRuntime
from video_analytics.config import load_config
from video_analytics.events.engine import EventEngine
from video_analytics.events.rules.line_crossing import (
    HorizontalLineCrossingRule,
)
from video_analytics.events.rules.zone_intrusion import (
    RectangularZoneIntrusionRule,
)
from video_analytics.events.rules.shared_aisle_warning import (
    SharedAisleWarningRule,
)
from video_analytics.inference.detector import ONNXDetector
from video_analytics.sources.video_source import VideoSource
from video_analytics.storage.database import Database
from video_analytics.storage.event_repository import EventRepository
from video_analytics.tracking.tracker import (
    ByteTrackAdapter,
    HIGH_CONF_DET_THRESHOLD,
)


@dataclass(frozen=True)
class ApplicationComponents:
    """
    Main Project 4 components assembled from configuration.
    """

    source: VideoSource
    detector: ONNXDetector
    tracker: ByteTrackAdapter
    database: Database
    repository: EventRepository
    event_engine: EventEngine
    event_processor: EventProcessor
    pipeline: VideoAnalyticsPipeline
    runtime: VideoAnalyticsRuntime


def build_application(
    config_path: str,
) -> ApplicationComponents:
    """
    Build the Project 4 application from a YAML configuration file.
    """

    config = load_config(config_path)

    source_config = config["source"]
    model_config = config["model"]
    tracking_config = config["tracking"]
    storage_config = config["storage"]
    rules_config = config["rules"]

    source = VideoSource(
        source_config["uri"]
    )

    source.open()

    try:
        source_fps = source.fps
    finally:
        source.release()

    if source_fps <= 0:
        source_fps = 30.0

    detector = ONNXDetector(
        model_path=model_config["path"],
        confidence_threshold=model_config[
            "confidence_threshold"
        ],
        nms_threshold=model_config[
            "nms_threshold"
        ],
    )

    # Pass both ByteTrack confidence thresholds explicitly.
    # Existing YAML files without high_conf_det_threshold
    # continue to use the previous default value (0.60).
    tracker = ByteTrackAdapter(
        track_activation_threshold=tracking_config[
            "track_activation_threshold"
        ],
        high_conf_det_threshold=tracking_config.get(
            "high_conf_det_threshold",
            HIGH_CONF_DET_THRESHOLD,
        ),
        minimum_consecutive_frames=tracking_config[
            "minimum_consecutive_frames"
        ],
        lost_track_buffer=tracking_config[
            "lost_track_buffer"
        ],
        frame_rate=source_fps,
    )

    database = Database(
        storage_config["database_path"]
    )

    database.initialize()

    repository = EventRepository(
        database
    )

    line_config = rules_config[
        "line_crossing"
    ]

    if line_config["enabled"]:
        line_rule = HorizontalLineCrossingRule(
            line_y=line_config["line_y"]
        )

        line_rule_id = line_config["id"]
    else:
        line_rule = None
        line_rule_id = None

    zone_config = rules_config[
        "zone_intrusion"
    ]

    if zone_config["enabled"]:
        zone_rule = RectangularZoneIntrusionRule(
            zone_id=zone_config["id"],
            x1=zone_config["x1"],
            y1=zone_config["y1"],
            x2=zone_config["x2"],
            y2=zone_config["y2"],
        )
    else:
        zone_rule = None

    # Opcional: configuracoes antigas continuam sem a nova regra.
    shared_config = rules_config.get("shared_aisle_warning", {})
    if shared_config.get("enabled", False):
        shared_aisle_rule = SharedAisleWarningRule(
            polygon_pixels=shared_config["polygon_pixels"],
            confirmation_frames=shared_config["confirmation_frames"],
            motion_window_frames=shared_config["motion_window_frames"],
            min_vehicle_displacement_px=shared_config["min_vehicle_displacement_px"],
            rearm_after_frames=shared_config["rearm_after_frames"],
        )
    else:
        shared_aisle_rule = None

    event_engine = EventEngine(
        line_rule=line_rule,
        source_id=source_config["id"],
        line_rule_id=line_rule_id,
        zone_rule=zone_rule,
        shared_aisle_rule=shared_aisle_rule,
        shared_aisle_rule_id=shared_config.get("id", "shared_aisle_warning"),
    )

    event_processor = EventProcessor(
        event_engine=event_engine,
        event_repository=repository,
    )

    pipeline = VideoAnalyticsPipeline(
        detector=detector,
        tracker=tracker,
        event_processor=event_processor,
    )

    runtime = VideoAnalyticsRuntime(
        source=source,
        pipeline=pipeline,
    )

    return ApplicationComponents(
        source=source,
        detector=detector,
        tracker=tracker,
        database=database,
        repository=repository,
        event_engine=event_engine,
        event_processor=event_processor,
        pipeline=pipeline,
        runtime=runtime,
    )