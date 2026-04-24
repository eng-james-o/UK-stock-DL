# UK-stock-DL

**UK-stock-DL** is a research-oriented project focused on predicting the UK stock market, specifically the FTSE-100 index, utilizing advanced deep learning techniques. The project aims to explore and implement various deep learning architectures to model and forecast stock market trends, providing insights into the efficacy of these models in financial time series prediction.

## Table of Contents

* [Overview](#overview)
* [Implemented Models](#implemented-models)
* [Planned Enhancements](#planned-enhancements)
* [Getting Started](#getting-started)
* [Project Structure](#project-structure)
* [License](#license)

## Overview

The volatility and complexity of financial markets present significant challenges for accurate forecasting. This project addresses these challenges by applying deep learning models to historical FTSE-100 index data, aiming to capture underlying patterns and trends. The primary objectives include:

* Evaluating the performance of various deep learning models in stock market prediction.
* Investigating the impact of hybrid models and attention mechanisms on forecasting accuracy.
* Laying the groundwork for integrating alternative data sources, such as news sentiment, into predictive models.

## Implemented Models

The project currently includes the following models:

* **CNN-LSTM Hybrid Model**: Combines Convolutional Neural Networks (CNN) for feature extraction with Long Short-Term Memory (LSTM) networks for sequence modeling.
* **GRU Model**: Utilizes Gated Recurrent Units (GRU) to capture temporal dependencies in stock price data.
* **GRU with Attention Mechanism**: Enhances the GRU model by incorporating an attention mechanism to focus on relevant time steps in the input sequence.

## Planned Enhancements

Future developments for the project include:

* **Generative Adversarial Networks (GANs)**: Implementing GANs to generate synthetic stock price data for training and evaluation purposes.
* **Transformer Models**: Applying transformer architectures to model long-range dependencies in time series data.
* **Hybrid RNN Architectures**: Exploring combinations of various Recurrent Neural Network (RNN) types to improve predictive performance.
* **Integration of Textual Data**: Mining and incorporating news feeds and other textual data sources to enrich model inputs with sentiment and contextual information.

## Getting Started

To replicate the experiments or build upon this work:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/eng-james-o/UK-stock-DL.git
   cd UK-stock-DL
   ```

2. **Set Up the Environment**:

   * Ensure you have Python 3.7 or higher installed.
   * Install the required packages:

     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Notebook**:

   * Open and execute the `dl-stock-prediction (1).ipynb` notebook to train and evaluate the models.

4. **Run Models and Pipelines via Python Scripts**:

   * You can run data download, preprocessing, feature engineering, model training, and evaluation directly from the modular Python scripts in the `src/` directory. Example usage:

   ```bash
   # Download data
   python src/data/fetch_data.py

   # Preprocess data
   python src/data/preprocess.py

   # Train models (CNN-LSTM, GRU, etc.)
   python src/models/train.py

   # Evaluate models
   python src/evaluation/evaluate.py
   ```

   * You may need to adapt the scripts or create a main pipeline script to orchestrate the workflow. See docstrings and comments in each module for details.

## Project Structure

```text
UK-stock-DL/
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # External datasets
├── notebooks/
│   └── dl-stock-prediction.ipynb  # Jupyter notebook for EDA and experiments
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── fetch_data.py    # Download data from yfinance
│   │   ├── preprocess.py    # Clean and process data, calculate indicators
│   ├── features/
│   │   └── technical_indicators.py # Feature engineering functions
│   ├── models/
│   │   ├── base_model.py    # Base model class
│   │   ├── model_cl.py      # CNN-LSTM model
│   │   ├── model_gru.py     # GRU model
│   │   └── train.py         # Training pipeline
│   ├── evaluation/
│   │   └── evaluate.py      # Evaluation & metrics
│   ├── utils/
│   │   └── helpers.py       # Utility scripts
│   └── plots/               # Plotting scripts (optional)
├── requirements.txt         # Python dependencies
├── LICENSE                  # Project license (Apache-2.0)
├── README.md                # Project documentation
```

## License

This project is licensed under the [Apache License 2.0](LICENSE), allowing for use, modification, and distribution under defined terms.
