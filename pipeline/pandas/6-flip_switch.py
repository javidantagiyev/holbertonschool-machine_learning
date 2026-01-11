#!/usr/bin/env python3
"""
6-flip_switch module.
"""

def flip_switch(df):
    """
    Sort in reverse chronological order and transpose.
    """
    return df.sort_index(ascending=False).T
