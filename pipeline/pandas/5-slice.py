#!/usr/bin/env python3
"""
5-slice module.
"""


def slice(df):
    """
    Extract High, Low, Close, and Volume_(BTC) columns and select every 60th.
    """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
