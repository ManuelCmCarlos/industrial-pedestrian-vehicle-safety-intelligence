from dataclasses import dataclass

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
MODEL_INPUT_SIZE = 640


CLASS_NAMES = [
    "person", "industrial_vehicle",
]


@dataclass(frozen=True)
class Detection:
    """
    One final object detection produced after confidence filtering
    and Non-Maximum Suppression.

    bbox uses absolute pixel coordinates:
    (x1, y1, x2, y2)
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


def postprocess_yolov8_output(
    output,
    original_width,
    original_height,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
):
    """
    Convert raw YOLOv8 ONNX output into final Detection objects.

    Expected raw output shape:
        (1, 84, 8400)

    The 84 values represent:
        4 bounding-box values + 80 COCO class scores.

    Processing:
        1. Remove the batch dimension.
        2. Transpose predictions to one row per candidate.
        3. Select the highest-scoring class.
        4. Remove weak candidates.
        5. Convert center-based boxes into top-left width/height format.
        6. Scale coordinates back to the original frame size.
        7. Apply Non-Maximum Suppression.
        8. Return structured Detection objects.
    """

    predictions = output[0].T

    boxes = []
    confidences = []
    class_ids = []

    for prediction in predictions:
        x_center, y_center, width, height = prediction[:4]
        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < confidence_threshold:
            continue

        x1 = x_center - width / 2
        y1 = y_center - height / 2

        scaled_x = int(x1 * original_width / MODEL_INPUT_SIZE)
        scaled_y = int(y1 * original_height / MODEL_INPUT_SIZE)
        scaled_width = int(width * original_width / MODEL_INPUT_SIZE)
        scaled_height = int(height * original_height / MODEL_INPUT_SIZE)

        boxes.append(
            [
                scaled_x,
                scaled_y,
                scaled_width,
                scaled_height,
            ]
        )
        confidences.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_threshold,
    )

    detections = []

    for index in indices.flatten():
        x, y, width, height = boxes[index]
        class_id = class_ids[index]

        detection = Detection(
            bbox=(
                x,
                y,
                x + width,
                y + height,
            ),
            confidence=confidences[index],
            class_id=class_id,
            class_name=CLASS_NAMES[class_id],
        )

        detections.append(detection)

    return detections