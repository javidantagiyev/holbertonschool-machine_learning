#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=["Weighted_Price"])

# Rename Timestamp -> Date and convert to datetime
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Index on Date
df = df.set_index("Date")

# Fill missing values
df["Close"] = df["Close"].fillna(method="ffill")
for col in ["High", "Low", "Open"]:
    df[col] = df[col].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

# 2017+ daily aggregation
df = df.loc["2017-01-01":].resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum",
})

# Print the transformed DataFrame (before plotting)
print(df)

# Plot
df.plot()
plt.show()
