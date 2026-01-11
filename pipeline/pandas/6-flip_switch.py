#!/usr/bin/env python3
"""
6-flip_switch module.

Sorts a DataFrame in reverse chronological order
and transposes it.
"""

import pandas as pd


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order
    and transpose it.

    Args:
        df: pandas.DataFrame containing a Timestamp column.

    Returns:
        pandas.DataFrame: transformed DataFrame.
    """
    # Sort in reverse chronological order (by index)
    df = df.sort_index(ascending=False)

    # Transpose the DataFrame
    return df.T
