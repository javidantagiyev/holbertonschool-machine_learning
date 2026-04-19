#!/usr/bin/env python3
"""
Adds two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): First array.
        arr2 (list): Second array.

    Returns:
        list or None: New list with summed values, or None if shapes differ.
    """
    # Arrays must have the same length
    if len(arr1) != len(arr2):
        return None

    return [a + b for a, b in zip(arr1, arr2)]
