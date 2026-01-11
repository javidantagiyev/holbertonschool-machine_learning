#!/usr/bin/env python3
"""
7-high module.

Sorts a DataFrame by the High price in descending order.
"""

import pandas as pd


def high(df):
    """
    Sort the DataFrame by the High column in descending order.

    Args:
        df: pandas.DataFrame containing a High column.

    Returns:
        pandas.DataFrame: sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
