#!/usr/bin/env python3
"""
Yolo v3 Object Detection Module
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class uses the YOLO v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t,
                 anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs

        Parameters:
        - outputs: list of numpy arrays (model predictions)
        - image_size: numpy array [image_height, image_width]

        Returns:
        - boxes, box_confidences, box_class_probs
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Extract components
            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            # Create grid
            cx = np.tile(np.arange(grid_w), grid_h)
            cx = cx.reshape(grid_h, grid_w)
            cy = np.tile(np.arange(grid_h), grid_w)
            cy = cy.reshape(grid_w, grid_h).T

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            # Calculate bx, by
            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_w
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_h

            # Calculate bw, bh
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

            # Convert to corner coords
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
