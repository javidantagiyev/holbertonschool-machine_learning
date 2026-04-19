#!/usr/bin/env python3
"""
Adds two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists or None: Result matrix or None if shapes differ.
    """
    # Check number of rows
    if len(mat1) != len(mat2):
        return None

    # Check number of columns per row
    if any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None

    return [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(mat1, mat2)]
