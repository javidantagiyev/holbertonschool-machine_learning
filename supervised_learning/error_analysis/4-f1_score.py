#!/usr/bin/env python3
"""Calculates the F1 score for each class in a confusion matrix"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class

    Parameters:
    confusion: numpy.ndarray of shape (classes, classes)
               where rows represent correct labels
               and columns represent predicted labels

    Returns:
    numpy.ndarray of shape (classes,)
    containing the F1 score of each class
    """
    # Get precision and recall (sensitivity)
    prec = precision(confusion)
    rec = sensitivity(confusion)

    # F1 score formula
    f1 = 2 * (prec * rec) / (prec + rec)

    return f1
