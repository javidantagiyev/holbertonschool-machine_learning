#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            new_row.append(sum(row[i] * mat2[i][j] for i in range(len(mat2))))
        result.append(new_row)
    return result
