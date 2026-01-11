#!/usr/bin/env python3
"""
10-index module.
"""


def index(df):
    """
    Set Timestamp column as the index of the DataFrame.
    """
    return df.set_index("Timestamp")
