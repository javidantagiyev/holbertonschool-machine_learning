#!/usr/bin/env python3
"""
Performs matrix multiplication using NumPy.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.

    Returns:
        numpy.ndarray: Product of the matrices.
    """
    return np.matmul(mat1, mat2)
