#!/usr/bin/env python3

import numpy as np
matrix = np.array([[36, 14, 57, 82, -9, 10], [100, 109, -36, 7, 2, 443], [1, 6, 23, 72, 12, 21], [5, 54, 10, 11, 16, 18]])
mat1 = matrix[1:3, :]
mat2 = matrix[:, 2:4]
mat3 = matrix[0:3, 3:6]
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
