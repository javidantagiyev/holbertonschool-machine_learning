#!/usr/bin/env python3
"""
7-high module.
"""

def high(df):
    """
    Sort the DataFrame by High column in descending order.
    """
    return df.sort_values(by="High", ascending=False)
