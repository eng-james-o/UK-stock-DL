import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU, BatchNormalization, Concatenate
from keras.optimizers import Adam
import numpy as np
from .base_model import BaseModel

class GANModel(BaseModel):
    """
    A Generative Adversarial Network (GAN) for stock price prediction.

    The GAN consists of two networks:
    1. Generator: An LSTM-based network that takes historical price sequences
       and technical indicators to predict the next closing price.
    2. Discriminator: A network that takes both the historical sequence and a price
       (either real from history or fake from the generator) and attempts to
       classify it as 'real' or 'fake'.

    Adversarial training helps the generator produce more realistic price predictions
    by trying to 'fool' the discriminator.
    """
    def __init__(self, seq_length=10, n_features=18, lookahead=1, learning_rate=0.0001):
        """
        Initialize the GAN model.

        Args:
            seq_length (int): Length of the input sequence.
            n_features (int): Number of features per time step.
            lookahead (int): Number of time steps to predict.
            learning_rate (float): Learning rate for both G and D.
        """
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.lookahead = lookahead
        self.learning_rate = learning_rate

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Build and compile the combined model (Generator + Frozen Discriminator)
        # This is used to train the Generator to fool the Discriminator
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.seq_length, self.n_features))
        generated_price = self.generator(gan_input)
        gan_output = self.discriminator([gan_input, generated_price])
        self.model = Model(gan_input, [generated_price, gan_output])

        # Compile models
        opt = Adam(learning_rate=self.learning_rate)

        # Discriminator is compiled first to be trainable
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Freeze Discriminator for GAN (combined) model training
        self.discriminator.trainable = False
        self.model.compile(
            loss=['mse', 'binary_crossentropy'],
            loss_weights=[100, 1], # Prioritize prediction accuracy (MSE) over fooling D
            optimizer=opt
        )

    def _build_generator(self):
        """
        Builds the Generator model using LSTM layers.

        Returns:
            keras.Model: The generator network.
        """
        model = Sequential(name="Generator")
        model.add(Input(shape=(self.seq_length, self.n_features)))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(LeakyReLU(negative_slope=0.2))
        model.add(Dense(self.lookahead, activation='sigmoid'))
        return model

    def _build_discriminator(self):
        """
        Builds the Discriminator model.
        It takes a sequence and a candidate price as inputs.

        Returns:
            keras.Model: The discriminator network.
        """
        input_seq = Input(shape=(self.seq_length, self.n_features))
        input_price = Input(shape=(self.lookahead,))

        # Process the temporal sequence
        x = LSTM(64)(input_seq)
        x = LeakyReLU(negative_slope=0.2)(x)

        # Concatenate the sequence representation with the target price
        combined = Concatenate(axis=1)([x, input_price])
        combined = Dense(64)(combined)
        combined = LeakyReLU(negative_slope=0.2)(combined)
        combined = Dense(32)(combined)
        combined = LeakyReLU(negative_slope=0.2)(combined)
        output = Dense(1, activation='sigmoid')(combined)

        return Model([input_seq, input_price], output, name="Discriminator")

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
        Adversarial training loop for the GAN.

        Args:
            X (np.ndarray): Input sequences.
            y (np.ndarray): Target prices.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.

        Returns:
            dict: Training history (D loss and G loss).
        """
        real_label = np.ones((batch_size, 1))
        fake_label = np.zeros((batch_size, 1))

        history = {'d_loss': [], 'g_loss': []}

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            idx = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[idx], y[idx]

            epoch_d_loss = []
            epoch_g_loss = []

            for i in range(0, len(X_shuffled) - batch_size, batch_size):
                X_batch = X_shuffled[i : i+batch_size]
                y_batch = y_shuffled[i : i+batch_size]

                # --- 1. Train Discriminator ---
                # Generate fake prices from current G
                generated_prices = self.generator.predict(X_batch, verbose=0)

                # Train D to recognize real and fake
                d_loss_real = self.discriminator.train_on_batch([X_batch, y_batch], real_label)
                d_loss_fake = self.discriminator.train_on_batch([X_batch, generated_prices], fake_label)

                # Average loss for reporting
                d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

                # --- 2. Train Generator (via Combined model) ---
                # Train G to produce prices that D thinks are real (real_label)
                g_loss = self.model.train_on_batch(X_batch, [y_batch, real_label])

                epoch_d_loss.append(d_loss)
                epoch_g_loss.append(g_loss[0])

            avg_d = np.mean(epoch_d_loss)
            avg_g = np.mean(epoch_g_loss)
            history['d_loss'].append(avg_d)
            history['g_loss'].append(avg_g)

            print(f"Epoch {epoch+1}/{epochs} [D loss: {avg_d:.4f}] [G loss: {avg_g:.4f}]")

        return history

    def predict(self, X, **kwargs):
        """
        Predict future prices using the Generator.
        """
        return self.generator.predict(X, **kwargs)

    def save(self, path):
        """
        Save the Generator model (the part used for prediction).
        """
        self.generator.save(path)

    def load(self, path):
        """
        Load a saved Generator model.
        """
        self.generator = load_model(path)

def build_gan(seq_length, n_features, lookahead):
    """
    Legacy wrapper for backward compatibility.
    Creates and returns the Generator model part of a GAN.
    """
    gan = GANModel(seq_length=seq_length, n_features=n_features, lookahead=lookahead)
    return gan.generator
