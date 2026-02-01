#!/usr/bin/env python3
"""
Calculates the shape of a matrix represented as nested Python lists.
"""


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list of integers.

    Args:
        matrix (list): A nested list representing a matrix.

    Returns:
        list: Shape of the matrix.
    """
    shape = []

    # Traverse through nested lists to determine dimensions
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
