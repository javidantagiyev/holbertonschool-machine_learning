#!/usr/bin/env python3
"""Randomly adjust image contrast"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Adjusts contrast randomly within [lower, upper]"""
    return tf.image.random_contrast(image, lower, upper)
