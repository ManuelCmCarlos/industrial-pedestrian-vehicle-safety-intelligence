import argparse


def parse_arguments():
    """
    Parse command-line arguments for the video analytics application.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Production-style real-time video analytics system."
        )
    )

    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help=(
            "Path to the YAML application configuration file. "
            "Default: config/default.yaml"
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional maximum number of frames to process. "
            "Useful for development and testing."
        ),
    )

    return parser.parse_args()


def main():
    """
    Build and run the configured video analytics application.
    """

    args = parse_arguments()

    from video_analytics.application.bootstrap import (
        build_application,
    )

    app = build_application(
        args.config
    )

    print(
        f"Source: {app.source.source}"
    )

    print(
        f"Tracker frame rate: "
        f"{app.tracker.frame_rate:.2f} FPS"
    )

    processed_frames = 0

    for runtime_result in app.runtime.run(
        max_frames=args.max_frames
    ):
        pipeline_result = (
            runtime_result.pipeline_result
        )

        source_timestamp = (
            runtime_result.source_timestamp
        )

        if source_timestamp is None:
            source_time_text = "unknown"
        else:
            source_time_text = (
                f"{source_timestamp:.3f}"
            )

        print(
            "frame="
            f"{runtime_result.frame_index} "
            "source_time="
            f"{source_time_text} "
            "processing_fps="
            f"{runtime_result.processing_fps:.2f} "
            "detections="
            f"{len(pipeline_result.detections)} "
            "tracks="
            f"{len(pipeline_result.tracked_objects)} "
            "events="
            f"{len(pipeline_result.persisted_events)}"
        )

        processed_frames += 1

    print(
        f"Processing complete. "
        f"Frames processed: {processed_frames}"
    )


if __name__ == "__main__":
    main()