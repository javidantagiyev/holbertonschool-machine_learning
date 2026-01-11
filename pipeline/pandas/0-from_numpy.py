#!/usr/bin/env python3
import pandas as pd
import string

def from_numpy(array):
    columns = list(string.ascii_uppercase[:array.shape[1]])
    return pd.DataFrame(array, columns=columns)
