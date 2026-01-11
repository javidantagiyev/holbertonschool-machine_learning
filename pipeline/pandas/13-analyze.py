#!/usr/bin/env python3
"""
13-analyze module.

Computes descriptive statistics for a DataFrame.
"""

import pandas as pd


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df: pandas.DataFrame containing a Timestamp column.

    Returns:
        pandas.DataFrame: DataFrame of descriptive statistics.
    """
    return df.drop(columns=["Timestamp"]).describe()
