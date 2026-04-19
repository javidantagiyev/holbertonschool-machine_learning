#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""

        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0 or k > self.n:
            return 0

        # Compute combination nCk
        factorial_n = 1
        for i in range(1, self.n + 1):
            factorial_n *= i

        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        factorial_nk = 1
        for i in range(1, self.n - k + 1):
            factorial_nk *= i

        combination = factorial_n / (factorial_k * factorial_nk)

        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""

        try:
            k = int(k)
        except Exception:
            return 0

        if k < 0:
            return 0

        if k >= self.n:
            return 1

        cumulative = 0
        for i in range(0, k + 1):
            cumulative += self.pmf(i)

        return cumulative
