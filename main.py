"""
main.py - Orchestrates the UK-stock-DL workflow: data download, preprocessing, feature engineering, model training, and evaluation.
"""
import os
from src.data.fetch_data import download_data
from src.data.preprocess import preprocess_data, normalize_data, lag_data
from src.features.technical_indicators import *
from src.models.model_cl import build_cnn_lstm_1d
from src.models.model_gru import build_gru, build_gru_attention
from src.models.base_model import BaseModel
from src.models.train import split_data
from src.evaluation.evaluate import MSE, MAPE, RMSE, SMAPE, R2, MASE, RMSSE, MDA, evaluate

import pandas as pd

# Configurations (could be loaded from a config file)
TICKER = 'FTSE'
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
RAW_DATA_PATH = 'data/raw/ftse.csv'
PROCESSED_DATA_PATH = 'data/processed/ftse_processed.csv'


def main():
    # 1. Download data
    if not os.path.exists(RAW_DATA_PATH):
        print('Downloading data...')
        data = download_data(TICKER, START_DATE, END_DATE)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
    else:
        print('Loading raw data...')
        data = pd.read_csv(RAW_DATA_PATH, index_col=0)

    # 2. Preprocess data
    print('Preprocessing data...')
    data = preprocess_data(data)
    data.to_csv(PROCESSED_DATA_PATH)

    # 3. Normalize and create sequences
    print('Normalizing and creating sequences...')
    data_norm, scaler = normalize_data(data)
    X, y = lag_data(data_norm)

    # 4. Split data
    print('Splitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Build and train model (example: CNN-LSTM 1D)
    print('Building and training model...')
    model = build_cnn_lstm_1d(X_train.shape[1], X_train.shape[2], y_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # 6. Evaluate model
    print('Evaluating model...')
    metrics = {'MSE': MSE, 'MAPE': MAPE, 'RMSE': RMSE, 'SMAPE': SMAPE, 'R2': R2, 'MASE': MASE, 'RMSSE': RMSSE, 'MDA': MDA}
    results = evaluate(model, metrics, X_test, y_test)
    print('Evaluation Results:', results)

if __name__ == '__main__':
    main()
