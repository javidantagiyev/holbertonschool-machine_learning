#!/usr/bin/env python3
"""Module for Yolo v3 object detection - Task 1: Process Outputs."""

import numpy as np
from tensorflow import keras


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo object detection model.

        Args:
            model_path (str): Path to the Darknet Keras model file.
            classes_path (str): Path to the file containing class names,
                listed in order of index.
            class_t (float): Box score threshold for the initial
                filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of shape
                (outputs, anchor_boxes, 2) containing all anchor boxes.
                outputs: number of outputs (predictions) by the model.
                anchor_boxes: number of anchor boxes per prediction.
                2 => [anchor_box_width, anchor_box_height].
        """
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the Darknet model for a single image.

        Args:
            outputs (list): List of numpy.ndarrays containing predictions
                from the Darknet model. Each output has shape:
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes).
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes.
            image_size (numpy.ndarray): Array containing the image's
                original size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                boxes: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 4) with
                    processed boundary boxes (x1, y1, x2, y2) relative
                    to the original image.
                box_confidences: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, 1) with
                    box confidences for each output.
                box_class_probs: list of numpy.ndarrays of shape
                    (grid_height, grid_width, anchor_boxes, classes) with
                    box class probabilities for each output.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        input_width = self.model.input.shape[2]
        input_height = self.model.input.shape[1]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract raw box parameters
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Build grid offsets c_x and c_y
            col = np.arange(grid_width).reshape(1, grid_width, 1)
            row = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_x = np.tile(col, (grid_height, 1, anchor_boxes))
            c_y = np.tile(row, (1, grid_width, anchor_boxes))

            # Anchor dimensions for this output scale
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Sigmoid activations for center offsets and confidence
            bx = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            by = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            # Exponential for width and height, normalized by input dims
            bw = (pw * np.exp(t_w)) / input_width
            bh = (ph * np.exp(t_h)) / input_height

            # Convert to corner format scaled to original image size
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Box confidence: sigmoid of raw value, shape (..., 1)
            conf_raw = output[..., 4:5]
            box_confidences.append(1 / (1 + np.exp(-conf_raw)))

            # Class probabilities: sigmoid of raw values
            class_raw = output[..., 5:]
            box_class_probs.append(1 / (1 + np.exp(-class_raw)))

        return boxes, box_confidences, box_class_probs
