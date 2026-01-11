#!/usr/bin/env python3
"""
1-from_dictionary module.

Creates a pandas DataFrame from a dictionary with labeled rows and columns.
"""

import pandas as pd

# Create the DataFrame
df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
    },
    index=["A", "B", "C", "D"]
)
