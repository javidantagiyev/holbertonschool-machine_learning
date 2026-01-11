#!/usr/bin/env python3
"""
12-hierarchy module.

Creates a concatenated DataFrame with a MultiIndex where Timestamp
is the first level and the exchange key is the second level.
"""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearrange MultiIndex so Timestamp is first level, then concatenate
    bitstamp and coinbase data for a specific time window in
    chronological order.

    Args:
        df1: pandas.DataFrame (coinbase)
        df2: pandas.DataFrame (bitstamp)

    Returns:
        pandas.DataFrame: concatenated DataFrame with Timestamp as first
        MultiIndex level.
    """
    # Index both DataFrames on Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Slice the required timestamp range (inclusive)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with keys (exchange as second level)
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Rearrange MultiIndex: Timestamp first, exchange second
    df = df.swaplevel(0, 1).sort_index(level=0)

    return df
