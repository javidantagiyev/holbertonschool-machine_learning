# Identity Block — Deep Residual Networks

## Overview

This module implements the **identity block** from the landmark paper
[Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385).

The identity block is the core building unit of ResNet architectures. It is
used when the input and output tensors share the same spatial dimensions and
depth, allowing the shortcut (skip) connection to bypass the convolution stack
and be added directly to the output.

---

## Architecture

```
input
  │
  ├──────────────────────────────────┐  (shortcut / identity path)
  │                                  │
  ▼                                  │
1×1 Conv (F11 filters) → BN → ReLU  │
  ▼                                  │
3×3 Conv (F3  filters) → BN → ReLU  │
  ▼                                  │
1×1 Conv (F12 filters) → BN         │
  │                                  │
  └──────────────── Add ─────────────┘
                     │
                   ReLU
                     │
                  output
```

---

## Function Signature

```python
def identity_block(A_prev, filters):
```

### Parameters

| Parameter | Type       | Description                                           |
| --------- | ---------- | ----------------------------------------------------- |
| `A_prev`  | Tensor     | Output from the previous layer (shortcut input)       |
| `filters` | tuple/list | `(F11, F3, F12)` — filter counts for each convolution |

- **F11** — filters in the first 1×1 convolution (dimension reduction)
- **F3** — filters in the 3×3 convolution (feature extraction)
- **F12** — filters in the second 1×1 convolution (dimension restoration)

### Returns

Activated output tensor of the identity block.

---

## Design Choices

| Detail                   | Value                                         |
| ------------------------ | --------------------------------------------- |
| Weight initializer       | He Normal (`seed=0`)                          |
| Batch normalization axis | `3` (channels-last)                           |
| Padding                  | `same` (preserves spatial dimensions)         |
| Activation               | ReLU (after each BN, and after the final Add) |

---

## Usage Example

```python
from tensorflow import keras as K
from 2-identity_block import identity_block   # noqa: F401

X = K.Input(shape=(224, 224, 256))
Y = identity_block(X, [64, 64, 256])
model = K.models.Model(inputs=X, outputs=Y)
model.summary()
```
