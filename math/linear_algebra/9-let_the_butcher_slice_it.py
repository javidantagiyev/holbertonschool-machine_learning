#!/usr/bin/env python3
"""
Demonstrates slicing on NumPy arrays.
"""
import numpy as np

matrix = np.array([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24]
])

# Middle two rows
mat1 = matrix[1:3, :]

# Middle two columns
mat2 = matrix[:, 2:4]

# Bottom-right 3x3 submatrix
mat3 = matrix[1:4, 3:6]

print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
