#!/usr/bin/env python3
"""Creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
    labels: one-hot numpy.ndarray of shape (m, classes)
            containing the correct labels
    logits: one-hot numpy.ndarray of shape (m, classes)
            containing the predicted labels

    Returns:
    confusion: numpy.ndarray of shape (classes, classes)
               where rows represent correct labels
               and columns represent predicted labels
    """
    # Get class indices from one-hot encoding
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]

    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))

    # Populate confusion matrix
    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
