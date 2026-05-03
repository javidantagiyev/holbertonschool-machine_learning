#!/usr/bin/env python3
"""
Yolo v3 Object Detection Module
"""

import tensorflow.keras as K


class Yolo:
    """
    Yolo class uses the YOLO v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Parameters:
        - model_path: path to the Darknet Keras model
        - classes_path: path to file containing class names
        - class_t: box score threshold for filtering
        - nms_t: IOU threshold for non-max suppression
        - anchors: numpy.ndarray of shape
                   (outputs, anchor_boxes, 2)
        """

        # Load the model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Store thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        