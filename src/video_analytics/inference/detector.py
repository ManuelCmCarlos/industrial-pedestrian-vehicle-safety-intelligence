from pathlib import Path

import numpy as np
import onnxruntime as ort

from video_analytics.inference.postprocessing import (
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    Detection,
    postprocess_yolov8_output,
)
from video_analytics.inference.preprocessing import preprocess_frame_for_onnx


class ONNXDetector:
    """
    YOLOv8 object detector backed by ONNX Runtime.

    This class owns the ONNX Runtime session and coordinates the
    preprocessing, inference, and postprocessing stages.

    It does not perform tracking, event processing, visualization,
    video capture, or persistent storage.
    """

    def __init__(
        self,
        model_path,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        providers=None,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame) -> list[Detection]:
        """
        Run object detection on one OpenCV frame.

        Args:
            frame:
                OpenCV BGR image.

        Returns:
            A list of postprocessed Detection objects.
        """

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Input frame must be a NumPy array."
            )

        original_height, original_width = frame.shape[:2]

        input_tensor = preprocess_frame_for_onnx(frame)

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor,
            },
        )

        detections = postprocess_yolov8_output(
            outputs[0],
            original_width=original_width,
            original_height=original_height,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

        return detections