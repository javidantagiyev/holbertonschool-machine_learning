#!/usr/bin/env python3
"""
3-rename module.

Renames the Timestamp column to Datetime and converts it to datetime format.
"""

import pandas as pd


def rename(df):
    """
    Rename Timestamp column to Datetime, convert it to datetime,
    and return only Datetime and Close columns.

    Args:
        df: pandas.DataFrame containing a Timestamp column.

    Returns:
        pandas.DataFrame: modified DataFrame.
    """
    # Rename column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert timestamp to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    # Return only required columns
    return df[["Datetime", "Close"]]
