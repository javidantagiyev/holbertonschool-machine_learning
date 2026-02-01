#!/usr/bin/env python3
"""
Transposes a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): Input matrix.

    Returns:
        list of lists: Transposed matrix.
    """
    # Swap rows and columns using list comprehension
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
