#!/usr/bin/env python3
"""
Concatenates two 2D matrices along a given axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.
        axis (int): Axis to concatenate on (0 or 1).

    Returns:
        list of lists or None: Concatenated matrix or None if invalid.
    """
    if axis == 0:
        # Same number of columns required
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # Same number of rows required
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]

    return None
