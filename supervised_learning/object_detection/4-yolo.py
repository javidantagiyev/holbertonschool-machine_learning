#!/usr/bin/env python3
"""Module for Yolo v3 object detection - Task 2: Filter Boxes."""

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter bounding boxes based on box score threshold.

        Computes the box score as the product of the box confidence and
        the highest class probability, then discards any box whose score
        falls below self.class_t.

        Args:
            boxes (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4) containing
                the processed boundary boxes for each output.
            box_confidences (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing
                the processed box confidences for each output.
            box_class_probs (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
                containing the processed box class probabilities for
                each output.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                filtered_boxes (numpy.ndarray): Shape (?, 4) containing
                    all filtered bounding boxes.
                box_classes (numpy.ndarray): Shape (?,) containing the
                    class index predicted by each filtered box.
                box_scores (numpy.ndarray): Shape (?,) containing the
                    box score for each filtered box.
        """
        all_filtered_boxes = []
        all_box_classes = []
        all_box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):

            # Compute per-class scores: confidence * class probability
            # confidence shape: (..., 1), class_probs shape: (..., classes)
            scores = confidence * class_probs

            # Best class index and score for each box
            box_classes = np.argmax(scores, axis=-1)
            box_scores = np.max(scores, axis=-1)

            # Build mask for boxes above the threshold
            mask = box_scores >= self.class_t

            # Apply mask to select boxes, classes, and scores
            all_filtered_boxes.append(box[mask])
            all_box_classes.append(box_classes[mask])
            all_box_scores.append(box_scores[mask])

        # Concatenate results across all output scales
        filtered_boxes = np.concatenate(all_filtered_boxes, axis=0)
        box_classes = np.concatenate(all_box_classes, axis=0)
        box_scores = np.concatenate(all_box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS)

        Parameters:
        - filtered_boxes: numpy.ndarray (?, 4)
        - box_classes: numpy.ndarray (?,)
        - box_scores: numpy.ndarray (?)

        Returns:
        - box_predictions: numpy.ndarray (?, 4)
        - predicted_box_classes: numpy.ndarray (?,)
        - predicted_box_scores: numpy.ndarray (?)
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            # Sort by score DESC
            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while cls_boxes.shape[0] > 0:
                # Take best box
                best_box = cls_boxes[0]
                best_score = cls_scores[0]

                box_predictions.append(best_box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(best_score)

                if cls_boxes.shape[0] == 1:
                    break

                rest_boxes = cls_boxes[1:]

                # Intersection
                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                # Areas
                best_area = ((best_box[2] - best_box[0]) *
                             (best_box[3] - best_box[1]))
                rest_areas = ((rest_boxes[:, 2] - rest_boxes[:, 0]) *
                              (rest_boxes[:, 3] - rest_boxes[:, 1]))

                union = best_area + rest_areas - intersection
                iou = intersection / union

                # Keep boxes with IoU < threshold
                keep = np.where(iou < self.nms_t)[0]

                cls_boxes = rest_boxes[keep]
                cls_scores = cls_scores[1:][keep]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a folder

        Parameters:
        - folder_path: path to folder containing images

        Returns:
        - images: list of numpy.ndarray images
        - image_paths: list of image file paths
        """

        import os
        import cv2

        images = []
        image_paths = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):
                image = cv2.imread(file_path)

                if image is not None:
                    images.append(image)
                    image_paths.append(file_path)

        return images, image_paths
