#!/usr/bin/env python3
"""
5-slice module.

Slices a DataFrame by selecting specific columns
and every 60th row.
"""

import pandas as pd


def slice(df):
    """
    Extract High, Low, Close, and Volume_(BTC) columns
    and select every 60th row.

    Args:
        df: pandas.DataFrame containing the required columns.

    Returns:
        pandas.DataFrame: sliced DataFrame.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
