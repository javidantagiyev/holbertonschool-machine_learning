#!/usr/bin/env python3
"""
Module that computes the sum of i squared from 1 to n
"""

def summation_i_squared(n):
    """Calculates sum of i^2 from 1 to n without using loops"""
    if not isinstance(n, int) or n < 1:
        return None

    return n * (n + 1) * (2 * n + 1) // 6