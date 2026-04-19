#!/usr/bin/env python3
"""Randomly change image brightness"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly changes brightness within [-max_delta, max_delta]"""
    return tf.image.random_brightness(image, max_delta)
