#!/usr/bin/env python3
"""
Transfer Learning module for CIFAR-10 classification using Keras applications.
Uses EfficientNetB0 as the base model with fine-tuning for CIFAR-10.
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data for the EfficientNetB0 model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 images.
        Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels.

    Returns:
        X_p: numpy.ndarray containing the preprocessed images.
        Y_p: numpy.ndarray containing the one-hot encoded labels.
    """
    X_p = K.applications.efficientnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load CIFAR-10 dataset
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Load EfficientNetB0 base model (pre-trained on ImageNet)
    # include_top=False removes the final classification layers
    base_model = K.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze all base model layers to preserve ImageNet features
    base_model.trainable = False

    # Build full model with Lambda layer to upscale 32x32 -> 224x224
    inputs = K.Input(shape=(32, 32, 3))

    # Scale up images from 32x32 to 224x224 (required by EfficientNetB0)
    x = K.layers.Lambda(
        lambda img: K.backend.resize_images(
            img,
            height_factor=7,
            width_factor=7,
            data_format='channels_last',
            interpolation='bilinear'
        ),
        name='upscale'
    )(inputs)

    # Pass through frozen base model (inference only, no training)
    x = base_model(x, training=False)

    # Add custom classification head
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(128, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)

    # Compile model for phase 1: train only the new head
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = K.callbacks.ModelCheckpoint(
        'cifar10.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    lr_scheduler = K.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
    early_stop = K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    # Phase 1: Train the classification head only
    print("=== Phase 1: Training classification head ===")
    model.fit(
        X_train_p, Y_train_p,
        batch_size=256,
        epochs=20,
        validation_data=(X_test_p, Y_test_p),
        callbacks=[checkpoint, lr_scheduler, early_stop]
    )

    # Phase 2: Fine-tune — unfreeze top layers of base model
    print("=== Phase 2: Fine-tuning top layers ===")
    base_model.trainable = True

    # Freeze all layers except the last 30 (top blocks of EfficientNetB0)
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with a lower learning rate for fine-tuning
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p, Y_train_p,
        batch_size=128,
        epochs=30,
        validation_data=(X_test_p, Y_test_p),
        callbacks=[checkpoint, lr_scheduler, early_stop]
    )

    print("Training complete. Best model saved as cifar10.h5")