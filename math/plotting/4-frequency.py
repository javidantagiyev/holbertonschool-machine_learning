#!/usr/bin/env python3
"""
4-frequency module.

Plots a histogram of student grades.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student scores for Project A.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))

    # bins every 10 units + outlined in black
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # labels/title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # force axis to match the reference look
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))

    plt.show()
