#!/usr/bin/env python3
"""
Yolo v3 Object Detection Module
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t,
                 nms_t, anchors):
        """Initialize Yolo"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """Process outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, _, _ = output.shape

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

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences,
                     box_class_probs):
        """Filter boxes"""
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

        return (np.concatenate(filtered_boxes),
                np.concatenate(box_classes),
                np.concatenate(box_scores))

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """Non-max suppression"""
        box_predictions = []
        pred_classes = []
        pred_scores = []

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
                pred_classes.append(cls)
                pred_scores.append(best_score)

                if cls_boxes.shape[0] == 1:
                    break

                rest = cls_boxes[1:]

                x1 = np.maximum(best_box[0], rest[:, 0])
                y1 = np.maximum(best_box[1], rest[:, 1])
                x2 = np.minimum(best_box[2], rest[:, 2])
                y2 = np.minimum(best_box[3], rest[:, 3])

                inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                area1 = ((best_box[2] - best_box[0]) *
                         (best_box[3] - best_box[1]))
                area2 = ((rest[:, 2] - rest[:, 0]) *
                         (rest[:, 3] - rest[:, 1]))

                iou = inter / (area1 + area2 - inter)

                keep = np.where(iou < self.nms_t)[0]

                cls_boxes = rest[keep]
                cls_scores = cls_scores[1:][keep]

        return (np.array(box_predictions),
                np.array(pred_classes),
                np.array(pred_scores))

    @staticmethod
    def load_images(folder_path):
        """Load images"""
        import os
        import cv2

        images = []
        paths = []

        for f in os.listdir(folder_path):
            path = os.path.join(folder_path, f)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    paths.append(path)

        return images, paths

    def preprocess_images(self, images):
        """Preprocess images"""
        import cv2

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        shapes = []

        for img in images:
            shapes.append([img.shape[0], img.shape[1]])

            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(shapes)

    def show_boxes(self, image, boxes, box_classes,
                   box_scores, file_name):
        """Show boxes"""
        import cv2
        import os

        img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)

            label = "{} {}".format(
                self.class_names[box_classes[i]],
                round(box_scores[i], 2)
            )

            y = y1 - 5 if y1 - 5 > 0 else y1 + 15

            cv2.putText(img, label, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, img)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            cv2.imwrite("detections/" + file_name, img)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Performs full YOLO prediction pipeline
        """

        images, image_paths = self.load_images(folder_path)

        predictions = []

        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        for i in range(len(images)):
            # Extract outputs per image
            img_outputs = [output[i] for output in outputs]

            boxes, box_conf, box_probs = self.process_outputs(
                img_outputs, image_shapes[i]
            )

            boxes, classes, scores = self.filter_boxes(
                boxes, box_conf, box_probs
            )

            boxes, classes, scores = self.non_max_suppression(
                boxes, classes, scores
            )

            predictions.append((boxes, classes, scores))

            # Show result
            file_name = image_paths[i].split('/')[-1]
            self.show_boxes(images[i], boxes, classes,
                            scores, file_name)

        return predictions, image_paths
