"""
main.py - Orchestrates the UK-stock-DL workflow: data download, news fetching, preprocessing, feature engineering, model training, and evaluation.
"""
import os
import pandas as pd
import numpy as np
from src.data.fetch_data import download_data, fetch_news
from src.features.sentiment import analyze_sentiment
from src.data.preprocess import preprocess_data, normalize_data, lag_data, unscale_data
from src.models.model_cl import CNNLSTMModel
from src.models.model_gru import GRUModel
from src.models.model_var import VARModel
from src.models.model_gan import GANModel
from src.models.train import split_data
from src.evaluation.evaluate import MSE, MAPE, RMSE, SMAPE, R2, MASE, RMSSE, MDA, evaluate
from src.utils.helpers import save_scaler, save_model
from src.utils.plotting import plot_predictions

# Configurations
TICKER = '^FTSE'
START_DATE = '2013-01-01'
END_DATE = '2023-10-01'
RAW_DATA_PATH = 'data/raw/ftse.csv'
PROCESSED_DATA_PATH = 'data/processed/ftse_processed.csv'
SCALER_PATH = 'models/scaler.joblib'
MODEL_PATH = 'models/gan_model.keras'

def main():
    # 1. Download stock data
    if not os.path.exists(RAW_DATA_PATH):
        print('Downloading stock data...')
        data = download_data(TICKER, START_DATE, END_DATE)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
    else:
        print('Loading raw stock data...')
        data = pd.read_csv(RAW_DATA_PATH, index_col=0, header=[0, 1])

    # Simplify columns if they are multi-index
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 2. Fetch and analyze news sentiment
    print('Fetching and analyzing news sentiment...')
    news_items = fetch_news(TICKER)
    sentiment_df = analyze_sentiment(news_items)

    # 3. Preprocess data
    print('Preprocessing data with technical indicators and sentiment...')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    data = preprocess_data(data, sentiment_df=sentiment_df)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH)

    # 4. Normalize and create sequences
    print('Normalizing and creating sequences...')
    data_norm, scaler = normalize_data(data)
    save_scaler(scaler, SCALER_PATH)

    X, y = lag_data(data_norm)

    # 5. Split data
    print('Splitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 6. Build and train GAN model
    print('Building and training GAN model...')
    gan_model = GANModel(seq_length=X_train.shape[1], n_features=X_train.shape[2])
    gan_model.fit(X_train, y_train, epochs=2, batch_size=32)
    save_model(gan_model, MODEL_PATH)

    # 7. Evaluate model
    print('Evaluating GAN model...')
    metrics = {'MSE': MSE, 'MAPE': MAPE, 'RMSE': RMSE, 'SMAPE': SMAPE, 'R2': R2, 'MASE': MASE, 'RMSSE': RMSSE, 'MDA': MDA}
    results = evaluate(gan_model, metrics, X_test, y_test)
    print('Evaluation Results:', results)

    # 8. Unscale and Plot results
    print('Plotting results...')
    y_test_unscaled = unscale_data(y_test.flatten(), scaler, col_idx=0)
    y_pred_unscaled = unscale_data(gan_model.predict(X_test).flatten(), scaler, col_idx=0)

    print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
