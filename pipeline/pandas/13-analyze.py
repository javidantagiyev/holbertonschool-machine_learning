#!/usr/bin/env python3
"""
13-analyze module.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.
    """
    return df.drop(columns=["Timestamp"]).describe()
