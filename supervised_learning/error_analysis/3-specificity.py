#!/usr/bin/env python3
"""Calculates the specificity for each class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class

    Parameters:
    confusion: numpy.ndarray of shape (classes, classes)
               where rows represent correct labels
               and columns represent predicted labels

    Returns:
    numpy.ndarray of shape (classes,)
    containing the specificity of each class
    """
    # True Positives
    true_pos = np.diag(confusion)

    # False Positives
    false_pos = np.sum(confusion, axis=0) - true_pos

    # False Negatives
    false_neg = np.sum(confusion, axis=1) - true_pos

    # True Negatives
    total = np.sum(confusion)
    true_neg = total - (true_pos + false_pos + false_neg)

    # Specificity = TN / (TN + FP)
    specificity = true_neg / (true_neg + false_pos)

    return specificity
