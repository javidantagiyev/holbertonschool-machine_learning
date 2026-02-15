#!/usr/bin/env python3
"""Posterior probability function"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for each probability in P
    given x successes out of n trials
    """

    # Validate n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Validate x
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Validate Pr
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    # Validate range of P
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError(
            "All values in P must be in the range [0, 1]"
        )

    # Validate range of Pr
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError(
            "All values in Pr must be in the range [0, 1]"
        )

    # Validate sum of Pr
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute binomial coefficient
    factorial = np.math.factorial
    comb = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))

    # Marginal probability
    marginal = np.sum(likelihood * Pr)

    # Posterior = Intersection / Marginal
    posterior = (likelihood * Pr) / marginal

    return posterior
