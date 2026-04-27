"""
main.py - Orchestrates the stock prediction workflow.
"""
import os
import pandas as pd
import numpy as np
import argparse
from src.data.fetch_data import download_data, fetch_news_sentiment
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

def main():
    parser = argparse.ArgumentParser(description='Stock Prediction Pipeline')
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'gan', 'cnnlstm', 'var'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--use_news', action='store_true', help='Include news sentiment analysis')
    args = parser.parse_args()

    # 1. Download data
    if not os.path.exists(RAW_DATA_PATH):
        print('Downloading data...')
        data = download_data(TICKER, START_DATE, END_DATE)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
    else:
        print('Loading raw data...')
        data = pd.read_csv(RAW_DATA_PATH, index_col=0, header=[0, 1, 2])
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    # 2. Fetch News Sentiment (Optional)
    sentiment_score = 0.0
    if args.use_news:
        print('Fetching news sentiment...')
        sentiment_score = fetch_news_sentiment(TICKER)
        print(f'Average Sentiment Score: {sentiment_score:.4f}')

    # 3. Preprocess data
    print('Preprocessing data...')
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    data = preprocess_data(data, sentiment_score=sentiment_score)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH)

    # 4. Normalize and create sequences
    print('Normalizing and creating sequences...')
    data_norm, scaler = normalize_data(data)
    os.makedirs('models', exist_ok=True)
    save_scaler(scaler, SCALER_PATH)

    X, y = lag_data(data_norm)

    # 5. Split data
    print('Splitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 6. Build and train model
    print(f'Building and training {args.model.upper()} model...')
    model_path = f'models/{args.model}_model.keras'

    if args.model == 'gru':
        model = GRUModel(use_attention=True, seq_length=X_train.shape[1], n_features=X_train.shape[2])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    elif args.model == 'gan':
        model = GANModel(seq_length=X_train.shape[1], n_features=X_train.shape[2])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, verbose=1)
    elif args.model == 'cnnlstm':
        model = CNNLSTMModel(seq_length=X_train.shape[1], n_features=X_train.shape[2])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    elif args.model == 'var':
        train_size = int(len(data_norm) * 0.8)
        var_train = data_norm.iloc[:train_size]
        model = VARModel(lags=4)
        model.fit(var_train)
        print("VAR model trained.")
        return

    save_model(model, model_path)

    # 7. Evaluate model
    print('Evaluating model...')
    metrics = {'MSE': MSE, 'MAPE': MAPE, 'RMSE': RMSE, 'SMAPE': SMAPE, 'R2': R2, 'MASE': MASE, 'RMSSE': RMSSE, 'MDA': MDA}
    results = evaluate(model, metrics, X_test, y_test)
    print('Evaluation Results:', results)

    print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
