#!/usr/bin/env python3
"""
9-fill module.

Cleans and fills missing values in a DataFrame.
"""

import pandas as pd


def fill(df):
    """
    Modify the DataFrame by handling missing values:
    - Remove the Weighted_Price column
    - Fill missing Close values with the previous row's value
    - Fill missing Open, High, and Low values with the Close value of the same row
    - Fill missing Volume_(BTC) and Volume_(Currency) values with 0

    Args:
        df: pandas.DataFrame to clean.

    Returns:
        pandas.DataFrame: cleaned DataFrame.
    """
    # Remove Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # Fill Close NaNs with previous value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill Open, High, Low NaNs with Close value
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    # Fill volume NaNs with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
