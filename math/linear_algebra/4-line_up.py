#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
