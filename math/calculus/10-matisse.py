#!/usr/bin/env python3
"""
Module that computes the derivative of a polynomial
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    return derivative if any(derivative) else [0]
