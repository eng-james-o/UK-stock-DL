# Project Structure

- `main.py`: Orchestrator for the end-to-end pipeline.
- `src/`: Source code directory.
  - `data/`: Data fetching and preprocessing.
    - `fetch_data.py`: Download stock prices and news sentiment.
    - `preprocess.py`: Cleaning, indicators, and normalization.
  - `features/`: Feature engineering.
    - `technical_indicators.py`: Classic trading indicators.
    - `sentiment.py`: Sentiment analysis logic.
  - `models/`: Machine learning architectures.
    - `base_model.py`: Interface for models.
    - `model_gan.py`: GAN implementation.
    - `model_gru.py`: GRU with Attention.
    - `model_cl.py`: CNN-LSTM.
    - `model_var.py`: VAR.
  - `evaluation/`: Metrics and evaluation scripts.
  - `utils/`: Helpers and plotting utilities.
- `tests/`: Unit and integration tests.
- `notebooks/`: Exploratory research notebooks.
