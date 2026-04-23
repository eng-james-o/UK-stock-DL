import unittest
import numpy as np
from src.models.model_gan import GANModel

class TestGANModel(unittest.TestCase):
    def setUp(self):
        self.seq_length = 5
        self.n_features = 3
        self.lookahead = 1
        self.model = GANModel(
            seq_length=self.seq_length,
            n_features=self.n_features,
            lookahead=self.lookahead
        )

    def test_initialization(self):
        self.assertEqual(self.model.seq_length, self.seq_length)
        self.assertEqual(self.model.n_features, self.n_features)
        self.assertIsNotNone(self.model.generator)
        self.assertIsNotNone(self.model.discriminator)

    def test_predict_shape(self):
        batch_size = 8
        X = np.random.rand(batch_size, self.seq_length, self.n_features).astype(np.float32)
        preds = self.model.predict(X, verbose=0)
        self.assertEqual(preds.shape, (batch_size, self.lookahead))

    def test_fit_mini(self):
        # Very small dataset for quick test
        batch_size = 4
        X = np.random.rand(batch_size * 4, self.seq_length, self.n_features).astype(np.float32)
        y = np.random.rand(batch_size * 4, self.lookahead).astype(np.float32)

        # Test 1 epoch
        history = self.model.fit(X, y, epochs=1, batch_size=batch_size)
        self.assertIn('d_loss', history)
        self.assertIn('g_loss', history)
        self.assertEqual(len(history['d_loss']), 1)

if __name__ == '__main__':
    unittest.main()
