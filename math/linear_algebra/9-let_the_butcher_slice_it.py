#!/usr/bin/env python3

import numpy as np
matrix = np.arange(1, 25).reshape(4, 6)
mat1 = matrix[1:3, :]
mat2 = matrix[:, 2:4]
mat3 = matrix[1:4, 3:6]
print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
