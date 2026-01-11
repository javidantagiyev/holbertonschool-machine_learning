#!/usr/bin/env python3
"""
2-from_file module.

Loads data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file as a pandas DataFrame.

    Args:
        filename: path to the file to load.
        delimiter: column separator used in the file.

    Returns:
        pandas.DataFrame: loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
