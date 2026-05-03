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
        """Processes Darknet outputs"""

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_conf = self.sigmoid(output[..., 4:5])
            class_probs = self.sigmoid(output[..., 5:])

            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h,
                                                            grid_w)
            cy = np.tile(np.arange(grid_h), grid_w).reshape(grid_w,
                                                            grid_h).T

            cx = cx[..., np.newaxis]
            cy = cy[..., np.newaxis]

            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_w
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            bw = (np.exp(t_wh[..., 0]) * anchor_w) / input_w
            bh = (np.exp(t_wh[..., 1]) * anchor_h) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)

            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences,
                     box_class_probs):
        """Filters boxes based on threshold"""

        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, conf, prob in zip(boxes, box_confidences,
                                box_class_probs):

            scores = conf * prob

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """Applies Non-Max Suppression"""

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while cls_boxes.shape[0] > 0:
                best_box = cls_boxes[0]
                best_score = cls_scores[0]

                box_predictions.append(best_box)
                predicted_box_classes.append(cls)
                predicted_box_scores.append(best_score)

                if cls_boxes.shape[0] == 1:
                    break

                rest_boxes = cls_boxes[1:]

                x1 = np.maximum(best_box[0], rest_boxes[:, 0])
                y1 = np.maximum(best_box[1], rest_boxes[:, 1])
                x2 = np.minimum(best_box[2], rest_boxes[:, 2])
                y2 = np.minimum(best_box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                best_area = ((best_box[2] - best_box[0]) *
                             (best_box[3] - best_box[1]))
                rest_areas = ((rest_boxes[:, 2] - rest_boxes[:, 0]) *
                              (rest_boxes[:, 3] - rest_boxes[:, 1]))

                union = best_area + rest_areas - intersection
                iou = intersection / union

                keep = np.where(iou < self.nms_t)[0]

                cls_boxes = rest_boxes[keep]
                cls_scores = cls_scores[1:][keep]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """Loads images from folder"""
        import os
        import cv2

        images = []
        image_paths = []

        for file_name in os.listdir(folder_path):
            path = os.path.join(folder_path, file_name)

            if os.path.isfile(path):
                image = cv2.imread(path)

                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for YOLO

        - Resize with cubic interpolation
        - Normalize pixel values to [0, 1]
        """

        import cv2

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])

            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)

            normalized = resized / 255.0

            pimages.append(normalized)

        return (np.array(pimages),
                np.array(image_shapes))
