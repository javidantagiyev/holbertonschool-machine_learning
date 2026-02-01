#!/usr/bin/env python3
"""
Concatenates NumPy arrays.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two NumPy arrays along a given axis.

    Args:
        mat1 (numpy.ndarray): First array.
        mat2 (numpy.ndarray): Second array.
        axis (int): Axis to concatenate on.

    Returns:
        numpy.ndarray: Concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
