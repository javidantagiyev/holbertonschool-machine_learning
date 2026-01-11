#!/usr/bin/env python3
"""
4-frequency module.

Plots a histogram of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student scores with bins every 10 units.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))

    # Histogram: bins every 10 units, outlined in black
    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, range=(0, 100), edgecolor='black')

    # Labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Match reference axis formatting
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))

    plt.show()
