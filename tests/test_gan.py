import numpy as np
import pytest
from src.models.model_gan import GANModel

def test_gan_initialization():
    model = GANModel(seq_length=5, n_features=3)
    assert model.generator is not None
    assert model.discriminator is not None

def test_gan_fit_predict():
    # Create dummy data
    X = np.random.rand(100, 5, 3)
    y = np.random.rand(100, 1)

    model = GANModel(seq_length=5, n_features=3)
    # Train for just 2 epochs for quick test
    model.fit(X, y, epochs=2, batch_size=10, verbose=0)

    # Predict
    preds = model.predict(X[:5])
    assert preds.shape == (5, 1)
    assert np.all(preds >= 0) and np.all(preds <= 1)
