#!/usr/bin/env python3
"""
11-concat module.

Indexes two DataFrames on Timestamp and concatenates them
with labeled keys.
"""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two DataFrames after indexing on Timestamp.

    - Index both DataFrames on Timestamp
    - Select all rows from df2 up to and including timestamp 1417411920
    - Concatenate df2 above df1
    - Label rows using keys: bitstamp and coinbase

    Args:
        df1: pandas.DataFrame (coinbase)
        df2: pandas.DataFrame (bitstamp)

    Returns:
        pandas.DataFrame: concatenated DataFrame.
    """
    # Index both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Select required rows from df2
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
