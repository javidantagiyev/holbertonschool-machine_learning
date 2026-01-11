#!/usr/bin/env python3
"""
9-fill module.
"""

def fill(df):
    """
    Clean and fill missing values according to the task requirements.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
