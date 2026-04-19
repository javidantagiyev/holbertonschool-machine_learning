#!/usr/bin/env python3
"""Calculates the precision for each class in a confusion matrix"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class

    Parameters:
    confusion: numpy.ndarray of shape (classes, classes)
               where rows represent correct labels
               and columns represent predicted labels

    Returns:
    numpy.ndarray of shape (classes,)
    containing the precision of each class
    """
    # True Positives = diagonal elements
    true_positives = np.diag(confusion)

    # False Positives = sum of column - true positive
    false_positives = np.sum(confusion, axis=0) - true_positives

    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives)

    return precision
