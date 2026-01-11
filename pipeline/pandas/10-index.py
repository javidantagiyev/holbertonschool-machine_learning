#!/usr/bin/env python3
"""
10-index module.

Sets the Timestamp column as the index of a DataFrame.
"""

import pandas as pd


def index(df):
    """
    Set the Timestamp column as the index of the DataFrame.

    Args:
        df: pandas.DataFrame containing a Timestamp column.

    Returns:
        pandas.DataFrame: DataFrame indexed by Timestamp.
    """
    return df.set_index("Timestamp")
