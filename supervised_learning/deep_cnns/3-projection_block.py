#!/usr/bin/env python3
"""Module for building a projection block in a ResNet architecture."""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block as described in Deep Residual Learning (2015).

    The projection block is used when the input and output dimensions differ
    (different spatial size or channel depth). A 1x1 convolution is applied
    to the shortcut path to project the input to the correct dimensions before
    adding it to the main path output.

    Args:
        A_prev: output tensor from the previous layer (the shortcut input).
        filters: tuple or list of (F11, F3, F12) where:
            - F11 (int): filters for the first 1x1 convolution.
            - F3  (int): filters for the 3x3 convolution.
            - F12 (int): filters for the second 1x1 convolution and the
              1x1 shortcut convolution.
        s (int): stride applied to the first convolution in the main path
            and to the 1x1 convolution in the shortcut path. Defaults to 2.

    Returns:
        Tensor: the activated output of the projection block after adding the
        projected shortcut connection and applying the final ReLU activation.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # --- Main path ---

    # First component: 1x1 conv with stride s (downsamples spatial dims)
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component: 3x3 conv
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component: 1x1 conv (no ReLU before shortcut addition)
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- Shortcut path ---
    # Project A_prev to match main path dimensions (stride s, F12 filters)
    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # --- Merge + final activation ---
    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
