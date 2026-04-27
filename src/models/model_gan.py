import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.optimizers import Adam
from .base_model import BaseModel

class GANModel(BaseModel):
    """
    Generative Adversarial Network (GAN) for stock price prediction.
    The Generator predicts the next price given a sequence.
    The Discriminator tries to distinguish real [sequence, price] pairs from generated ones.
    """
    def __init__(self, seq_length=10, n_features=18, latent_dim=1):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Discriminator is compiled separately
        self.discriminator.compile(optimizer=Adam(learning_rate=0.0001),
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])

        # Combined model for training the generator
        # During combined model training, we only update the generator
        self.discriminator.trainable = False

        input_seq = Input(shape=(self.seq_length, self.n_features))
        generated_price = self.generator(input_seq)
        validity = self.discriminator([input_seq, generated_price])

        self.combined = Model(input_seq, validity)
        self.combined.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

    def _build_generator(self):
        """Builds an LSTM-based generator."""
        model = Sequential(name="Generator")
        model.add(LSTM(64, return_sequences=True, input_shape=(self.seq_length, self.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(self.latent_dim, activation='sigmoid'))
        return model

    def _build_discriminator(self):
        """Builds a discriminator that takes a sequence and a price value."""
        input_seq = Input(shape=(self.seq_length, self.n_features))
        input_price = Input(shape=(self.latent_dim,))

        # Process sequence with LSTM
        x = LSTM(32)(input_seq)

        # Combine sequence representation with the price to evaluate
        combined = Concatenate()([x, input_price])

        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        validity = Dense(1, activation='sigmoid')(x)

        return Model([input_seq, input_price], validity, name="Discriminator")

    def fit(self, X, y, epochs=10, batch_size=32, verbose=1, **kwargs):
        """Custom training loop for the GAN."""
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of data
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_seqs = X[idx]
            real_prices = y[idx]

            # Generate fake prices
            generated_prices = self.generator.predict(real_seqs, verbose=0)

            # Train Discriminator
            d_loss_real = self.discriminator.train_on_batch([real_seqs, real_prices], valid)
            d_loss_fake = self.discriminator.train_on_batch([real_seqs, generated_prices], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(real_seqs, valid)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    def predict(self, X, **kwargs):
        """Predict using the generator."""
        return self.generator.predict(X, **kwargs)

    def save(self, path):
        """Save the generator model."""
        self.generator.save(path)

    def load(self, path):
        """Load the generator model."""
        self.generator = load_model(path)

def build_gan(seq_length, n_features, latent_dim=1):
    """Utility function to create a GANModel instance."""
    return GANModel(seq_length, n_features, latent_dim)
