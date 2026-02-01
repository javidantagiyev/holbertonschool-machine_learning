#!/usr/bin/env python3
"""
Performs element-wise operations using NumPy.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction,
    multiplication, and division.

    Args:
        mat1 (numpy.ndarray): First input.
        mat2 (numpy.ndarray or scalar): Second input.

    Returns:
        tuple: (add, sub, mul, div)
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
