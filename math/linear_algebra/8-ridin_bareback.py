#!/usr/bin/env python3
"""
Performs matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists or None: Product matrix or None if invalid.
    """
    # Number of columns in mat1 must equal rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    result = []

    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            # Dot product
            new_row.append(sum(row[i] * mat2[i][j] for i in range(len(mat2))))
        result.append(new_row)

    return result
