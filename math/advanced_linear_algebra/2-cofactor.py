#!/usr/bin/env python3
"""
Module that computes the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    sub.append(matrix[r][:j] + matrix[r][j + 1:])
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * _determinant(sub))
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix


def _determinant(matrix):
    """
    Helper function to calculate determinant recursively
    """
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - \
               matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = []
        for row in matrix[1:]:
            sub.append(row[:col] + row[col + 1:])
        det += ((-1) ** col) * matrix[0][col] * _determinant(sub)

    return det
