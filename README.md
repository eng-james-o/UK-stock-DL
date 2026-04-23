# UK-stock-DL

**UK-stock-DL** is a research-oriented project focused on predicting the UK stock market, specifically the FTSE-100 index, utilizing advanced deep learning techniques. The project aims to explore and implement various deep learning architectures to model and forecast stock market trends, providing insights into the efficacy of these models in financial time series prediction.

## Table of Contents

* [Overview](#overview)
* [Implemented Models](#implemented-models)
* [Features & News Integration](#features--news-integration)
* [Getting Started](#getting-started)
* [Project Structure](#project-structure)
* [License](#license)

## Overview

The volatility and complexity of financial markets present significant challenges for accurate forecasting. This project addresses these challenges by applying deep learning models to historical FTSE-100 index data, aiming to capture underlying patterns and trends.

## Implemented Models

The project currently includes the following models:

* **GAN (Generative Adversarial Network)**: Uses a Generator to forecast prices and a Discriminator to improve accuracy through adversarial training.
* **CNN-LSTM Hybrid Model**: Combines CNNs for feature extraction with LSTMs for sequence modeling.
* **GRU Model**: Utilizes Gated Recurrent Units (GRU) to capture temporal dependencies, with optional Attention mechanisms.
* **VAR Model**: A classical multivariate statistical model.

## Features & News Integration

In addition to technical indicators (RSI, ATR, etc.), this project integrates **alternative data**:
- **News Sentiment**: Real-time news headlines are fetched via `yfinance` and analyzed using the `VADER` sentiment analyzer to provide context to price movements.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/eng-james-o/UK-stock-DL.git
   cd UK-stock-DL
   ```

2. **Set Up the Environment**:
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Pipeline**:
   - The main entry point orchestrates the entire workflow:
     ```bash
     python main.py
     ```

## Project Structure

- `src/data/`: Data acquisition and preprocessing.
- `src/features/`: Technical indicators and sentiment analysis.
- `src/models/`: Deep learning model architectures (GAN, GRU, etc.).
- `src/evaluation/`: Performance metrics and evaluation logic.
- `src/utils/`: Helpers for plotting and persistence.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
