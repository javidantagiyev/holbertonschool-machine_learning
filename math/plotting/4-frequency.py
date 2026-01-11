#!/usr/bin/env python3
"""
4-frequency module.

Plots a histogram of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student scores.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))

    # Histogram: bins every 10 units, black edges
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Force axes to match reference
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))
    plt.ylim(0, 30)

    plt.show()
