#!/usr/bin/env python3
"""
0-from_numpy module.

Creates a pandas DataFrame from a NumPy ndarray with columns labeled
in alphabetical order (A-Z).
"""

import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy ndarray.

    The DataFrame columns are labeled in alphabetical order and capitalized.
    There will not be more than 26 columns.

    Args:
        array: numpy.ndarray to convert into a DataFrame.

    Returns:
        pandas.DataFrame: newly created DataFrame.
    """
    columns = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
