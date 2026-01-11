#!/usr/bin/env python3
"""
6-bars module.

Plots a stacked bar graph representing fruit quantities per person.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar graph of fruit quantities per person.

    Requirements:
    - bars grouped by person: Farrah, Fred, Felicia
    - stacked order bottom->top: apples, bananas, oranges, peaches
    - colors:
        apples: red
        bananas: yellow
        oranges: #ff8000
        peaches: #ffe5b4
    - legend indicates fruit colors
    - bar width: 0.5
    - y-axis label: Quantity of Fruit
    - y-axis range: 0 to 80, ticks every 10
    - title: Number of Fruit per Person
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    a = apples
    b = apples + bananas
    c = apples + bananas + oranges

    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(x, bananas, width=0.5, bottom=a, color='yellow', label='bananas')
    plt.bar(x, oranges, width=0.5, bottom=b, color='#ff8000', label='oranges')
    plt.bar(x, peaches, width=0.5, bottom=c, color='#ffe5b4', label='peaches')

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()

    plt.show()
