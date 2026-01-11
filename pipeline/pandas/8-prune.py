#!/usr/bin/env python3
"""
8-prune module.
"""

def prune(df):
    """
    Remove rows where Close is NaN.
    """
    return df.dropna(subset=["Close"])
