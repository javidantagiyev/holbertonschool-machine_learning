#!/usr/bin/env python3
"""Module for building the ResNet-50 architecture."""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Build the ResNet-50 architecture (He et al., 2015).

    The network consists of an initial convolution + pooling stage followed
    by four groups of residual blocks (conv2_x through conv5_x), an average
    pooling layer, and a fully-connected softmax output layer.

    Block counts per stage (50-layer column of the paper's Table 1):
        - conv2_x : 1 projection block  + 2 identity blocks  (3 total)
        - conv3_x : 1 projection block  + 3 identity blocks  (4 total)
        - conv4_x : 1 projection block  + 5 identity blocks  (6 total)
        - conv5_x : 1 projection block  + 2 identity blocks  (3 total)

    Input shape assumed: (224, 224, 3).
    All convolutions use He Normal initialisation (seed=0) and are followed
    by Batch Normalisation (channels axis) then ReLU activation.

    Returns:
        keras.Model: compiled Keras model representing ResNet-50.
    """
    init = K.initializers.HeNormal(seed=0)

    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1 – conv1: 7×7, 64 filters, stride 2  →  MaxPool 3×3 stride 2
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=init
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Stage 2 – conv2_x: 3 blocks, filters [64, 64, 256], stride=1
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3 – conv3_x: 4 blocks, filters [128, 128, 512], stride=2
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4 – conv4_x: 6 blocks, filters [256, 256, 1024], stride=2
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5 – conv5_x: 3 blocks, filters [512, 512, 2048], stride=2
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Output – Average Pooling  →  Fully-connected (1000 classes, softmax)
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='same'
    )(X)
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
    )(X)

    model = K.models.Model(inputs=X_input, outputs=X)

    return model
