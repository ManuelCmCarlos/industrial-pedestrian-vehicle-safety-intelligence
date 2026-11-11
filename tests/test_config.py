from pathlib import Path

import pytest

from video_analytics.config import load_config


def test_load_config_reads_expected_sections():
    config = load_config(
        "config/default.yaml"
    )

    assert set(config.keys()) == {
        "source",
        "model",
        "tracking",
        "storage",
        "rules",
    }

    assert config["source"]["id"] == "camera_01"

    assert (
        config["model"]["path"]
        == "outputs/training/baseline_02/weights/best.onnx"
    )

    assert (
        config["tracking"][
            "track_activation_threshold"
        ]
        == 0.60
    )

    assert (
        config["storage"]["database_path"]
        == "data/events.db"
    )

    assert (
        config["rules"]["line_crossing"][
            "enabled"
        ]
        is True
    )

    assert (
        config["rules"]["zone_intrusion"]["id"]
        == "restricted_zone"
    )


def test_load_config_missing_file_raises_error():
    missing_path = Path(
        "config/this_file_does_not_exist.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Configuration file not found",
    ):
        load_config(missing_path)