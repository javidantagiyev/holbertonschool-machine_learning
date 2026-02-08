#!/usr/bin/env python3
"""
Module that determines the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2:
        return None

    n, m = matrix.shape
    if n == 0 or n != m:
        return None

    if not np.array_equal(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None

    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
