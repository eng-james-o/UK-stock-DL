# UK-stock-DL

**UK-stock-DL** is a research-oriented project focused on predicting the UK stock market, specifically the FTSE-100 index, utilizing advanced deep learning techniques. The project aims to explore and implement various deep learning architectures to model and forecast stock market trends, providing insights into the efficacy of these models in financial time series prediction.

## Table of Contents

* [Overview](#overview)
* [Implemented Models](#implemented-models)
* [Planned Enhancements](#planned-enhancements)
* [Getting Started](#getting-started)
* [Project Structure](#project-structure)
* [License](#license)
* [Acknowledgements](#acknowledgements)

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

## Project Structure

```
UK-stock-DL/
├── dl-stock-prediction (1).ipynb  # Jupyter notebook containing model implementations and experiments
├── requirements.txt               # List of required Python packages
├── LICENSE                        # Project license (Apache-2.0)
└── README.md                      # Project documentation
```

## License

This project is licensed under the [Apache License 2.0](LICENSE), allowing for use, modification, and distribution under defined terms.
