#!/usr/bin/env python3
"""
4-array module.

Selects the last 10 rows of High and Close columns
and converts them to a NumPy array.
"""

import pandas as pd


def array(df):
    """
    Convert the last 10 rows of High and Close columns to a NumPy array.

    Args:
        df: pandas.DataFrame containing High and Close columns.

    Returns:
        numpy.ndarray: array of selected values.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
