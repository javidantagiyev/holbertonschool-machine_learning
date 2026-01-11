#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plot exponential decay with a logarithmic y-axis.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot line graph
    plt.plot(x, y)

    # Axis labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Logarithmic y-axis
    plt.yscale('log')

    # X-axis range
    plt.xlim(0, 28650)

    plt.show()
