from video_analytics.application import bootstrap


class FakeVideoSource:
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.open_called = False
        self.release_called = False

    @property
    def fps(self):
        return 60.0

    def open(self):
        self.open_called = True
        self.capture = object()

    def release(self):
        self.release_called = True
        self.capture = None


class FakeDetector:
    def __init__(
        self,
        model_path,
        confidence_threshold,
        nms_threshold,
    ):
        self.model_path = model_path
        self.confidence_threshold = (
            confidence_threshold
        )
        self.nms_threshold = nms_threshold


class FakeTracker:
    def __init__(
        self,
        track_activation_threshold,
        high_conf_det_threshold,
        minimum_consecutive_frames,
        lost_track_buffer,
        frame_rate,
    ):
        self.track_activation_threshold = (
            track_activation_threshold
        )
        self.high_conf_det_threshold = ( 
            high_conf_det_threshold
        )    
        self.minimum_consecutive_frames = (
            minimum_consecutive_frames
        )
        self.lost_track_buffer = (
            lost_track_buffer
        )
        self.frame_rate = frame_rate


class FakeDatabase:
    def __init__(self, database_path):
        self.database_path = database_path
        self.initialize_called = False

    def initialize(self):
        self.initialize_called = True


def test_build_application_uses_configuration(
    monkeypatch,
):
    created_sources = []

    def fake_source_factory(source):
        fake_source = FakeVideoSource(
            source
        )

        created_sources.append(
            fake_source
        )

        return fake_source

    monkeypatch.setattr(
        bootstrap,
        "VideoSource",
        fake_source_factory,
    )

    monkeypatch.setattr(
        bootstrap,
        "ONNXDetector",
        FakeDetector,
    )

    monkeypatch.setattr(
        bootstrap,
        "ByteTrackAdapter",
        FakeTracker,
    )

    monkeypatch.setattr(
        bootstrap,
        "Database",
        FakeDatabase,
    )

    app = bootstrap.build_application(
        "config/default.yaml"
    )

    source = created_sources[0]

    assert source.source == (
        "../project_3_cv_deployment/"
        "videos/input.mp4"
    )

    assert source.open_called is True
    assert source.release_called is True
    assert source.capture is None

    assert app.detector.model_path == (
        "outputs/training/baseline_02/weights/best.onnx"
    )

    assert (
        app.detector.confidence_threshold
        == 0.25
    )

    assert app.detector.nms_threshold == 0.45

    assert (
        app.tracker.track_activation_threshold
        == 0.60
    )

    assert (
        app.tracker.minimum_consecutive_frames
        == 2
    )

    assert app.tracker.lost_track_buffer == 30

    assert app.tracker.frame_rate == 60.0

    assert (
        app.database.database_path
        == "data/events.db"
    )

    assert (
        app.database.initialize_called
        is True
    )

    assert app.event_engine.line_rule is not None

    assert (
        app.event_engine.line_rule.line_y
        == 750
    )

    assert (
        app.event_engine.line_rule_id
        == "entrance_line"
    )

    assert app.event_engine.zone_rule is not None

    assert (
        app.event_engine.zone_rule.zone_id
        == "restricted_zone"
    )

    assert (
        app.event_engine.source_id
        == "camera_01"
    )

    assert app.source is source

    assert (
        app.pipeline.detector
        is app.detector
    )

    assert (
        app.pipeline.tracker
        is app.tracker
    )

    assert (
        app.runtime.pipeline
        is app.pipeline
    )

    assert (
        app.runtime.source
        is app.source
    )