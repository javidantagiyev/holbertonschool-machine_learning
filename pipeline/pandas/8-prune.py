#!/usr/bin/env python3
"""
8-prune module.

Removes rows where the Close value is NaN.
"""

import pandas as pd


def prune(df):
    """
    Remove entries where Close has NaN values.

    Args:
        df: pandas.DataFrame containing a Close column.

    Returns:
        pandas.DataFrame: DataFrame without NaN Close values.
    """
    return df.dropna(subset=["Close"])
