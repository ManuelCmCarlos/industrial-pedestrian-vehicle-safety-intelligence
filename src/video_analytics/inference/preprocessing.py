import cv2
import numpy as np


MODEL_INPUT_SIZE = 640


def preprocess_frame_for_onnx(frame):
    """
    Convert one OpenCV frame into the tensor format expected
    by the YOLOv8 ONNX model.

    Input:
        OpenCV image in BGR format with shape:
        (height, width, channels)

    Output:
        NumPy float32 tensor with shape:
        (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

    Processing:
        1. Resize the image to the model input size.
        2. Convert BGR to RGB.
        3. Normalize pixel values from 0-255 to 0.0-1.0.
        4. Convert HWC layout to CHW layout.
        5. Add the batch dimension.
    """

    resized_frame = cv2.resize(
        frame,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
    )

    rgb_frame = cv2.cvtColor(
        resized_frame,
        cv2.COLOR_BGR2RGB,
    )

    normalized_frame = rgb_frame.astype(np.float32) / 255.0

    input_tensor = np.transpose(
        normalized_frame,
        (2, 0, 1),
    )

    input_tensor = np.expand_dims(
        input_tensor,
        axis=0,
    )

    return input_tensor.astype(np.float32)