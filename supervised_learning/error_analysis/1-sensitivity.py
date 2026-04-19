#!/usr/bin/env python3
"""Calculates the sensitivity for each class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class

    Parameters:
    confusion: numpy.ndarray of shape (classes, classes)
               where rows represent correct labels
               and columns represent predicted labels

    Returns:
    numpy.ndarray of shape (classes,)
    containing the sensitivity of each class
    """
    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # False Negatives = sum of row - true positive
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Sensitivity = TP / (TP + FN)
    sensitivity = true_positives / (true_positives + false_negatives)

    return sensitivity
