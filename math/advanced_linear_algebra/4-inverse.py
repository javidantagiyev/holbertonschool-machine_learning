#!/usr/bin/env python3
"""
Module that computes the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix
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
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    det = _determinant(matrix)
    if det == 0:
        return None

    adj = _adjugate(matrix)

    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)

    return inv


def _adjugate(matrix):
    """
    Computes the adjugate matrix
    """
    cof = _cofactor(matrix)
    n = len(cof)

    adj = []
    for i in range(n):
        adj_row = []
        for j in range(n):
            adj_row.append(cof[j][i])
        adj.append(adj_row)

    return adj


def _cofactor(matrix):
    """
    Computes the cofactor matrix
    """
    n = len(matrix)
    cof = []

    for i in range(n):
        row_cof = []
        for j in range(n):
            sub = []
            for r in range(n):
                if r != i:
                    sub.append(matrix[r][:j] + matrix[r][j + 1:])
            sign = (-1) ** (i + j)
            row_cof.append(sign * _determinant(sub))
        cof.append(row_cof)

    return cof


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
