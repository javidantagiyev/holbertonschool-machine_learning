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
    true_positives = np.diag(confusion)

    # False Positives
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False Negatives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # True Negatives
    total = np.sum(confusion)
    true_negatives = total - (true_positives + false_positives + false_negatives)

    # Specificity = TN / (TN + FP)
    specificity = true_negatives / (true_negatives + false_positives)

    return specificity
