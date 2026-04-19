#!/usr/bin/env python3
"""
Module that computes the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if (not isinstance(poly, list) or len(poly) == 0 or
            not isinstance(C, int)):
        return None

    if poly == [0]:
        return [C]

    integral = [C]

    for i in range(len(poly)):
        coef = poly[i] / (i + 1)
        if coef.is_integer():
            coef = int(coef)
        integral.append(coef)

    # remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
