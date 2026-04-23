# Models

This directory contains the implementations of various deep learning and statistical models used for stock price prediction.

## Model Architectures

### 1. GAN (Generative Adversarial Network)
- **File**: `model_gan.py`
- **Description**: A Generative Adversarial Network where the Generator predicts future prices and the Discriminator tries to distinguish them from actual historical movements.
- **Key Features**: LSTM-based Generator, custom adversarial training loop.

### 2. GRU (Gated Recurrent Unit)
- **File**: `model_gru.py`
- **Description**: A GRU-based model with an optional Attention mechanism.
- **Key Features**: Supports attention-based weighting of temporal features.

### 3. CNN-LSTM
- **File**: `model_cl.py`
- **Description**: A hybrid architecture combining Convolutional Neural Networks for spatial feature extraction and LSTMs for temporal modeling.

### 4. VAR (Vector Autoregression)
- **File**: `model_var.py`
- **Description**: A classical statistical model for multivariate time-series forecasting.

## Base Class

All models inherit from `BaseModel` defined in `base_model.py`, ensuring a consistent interface:
- `fit(X, y, **kwargs)`: Train the model.
- `predict(X, **kwargs)`: Generate predictions.
- `save(path)`: Persist the model to disk.
- `load(path)`: Load a saved model.

### Detailed Architecture: GAN
The `GANModel` implements an adversarial approach to time-series forecasting.

**Generator**:
- Inputs: `(batch_size, seq_length, n_features)`
- Layers: 2x LSTM layers (64, 128 units) with Dropout, followed by Dense layers with LeakyReLU.
- Output: `(batch_size, lookahead)` representing the normalized predicted price.

**Discriminator**:
- Inputs: A pair of `(sequence, price)`.
- Layers: LSTM layer for the sequence, Concatenation with the price, followed by multiple Dense layers.
- Output: A single probability value (0 to 1) indicating whether the input price is 'real' (matches historical future) or 'fake' (produced by G).

**Training Strategy**:
- Loss: MSE for price accuracy and Binary Crossentropy for adversarial feedback.
- Optimization: Adam optimizer with a low learning rate (default 0.0001) to ensure stable adversarial training.
