#!/usr/bin/env python3
"""
4-array module.
"""


def array(df):
    """
    Convert the last 10 rows of High and Close columns to a NumPy array.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
