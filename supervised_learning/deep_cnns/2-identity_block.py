#!/usr/bin/env python3
"""Module for building an identity block in a ResNet architecture."""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block as described in Deep Residual Learning (2015).

    The identity block is the standard block used in ResNets, and corresponds
    to the case where the input activation has the same dimension as the
    output activation. It applies three convolutions (1x1, 3x3, 1x1) with
    batch normalization and ReLU activations, then adds the shortcut (input)
    directly to the output before the final activation.

    Args:
        A_prev: output tensor from the previous layer (the shortcut input).
        filters: tuple or list of (F11, F3, F12) where:
            - F11 (int): number of filters for the first 1x1 convolution.
            - F3  (int): number of filters for the 3x3 convolution.
            - F12 (int): number of filters for the second 1x1 convolution.

    Returns:
        Tensor: the activated output of the identity block after adding the
        shortcut connection and applying the final ReLU activation.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # --- First component: 1x1 conv ---
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Second component: 3x3 conv ---
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # --- Third component: 1x1 conv (no ReLU before shortcut addition) ---
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # --- Shortcut connection + final activation ---
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
