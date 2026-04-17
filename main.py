"""
main.py - Orchestrates the UK-stock-DL workflow: data download, preprocessing, feature engineering, model training, and evaluation.
"""
import os
import pandas as pd
import numpy as np
from src.data.fetch_data import download_data
from src.data.preprocess import preprocess_data, normalize_data, lag_data, unscale_data
from src.models.model_cl import CNNLSTMModel
from src.models.model_gru import GRUModel
from src.models.model_var import VARModel
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
MODEL_PATH = 'models/gru_model.keras'

def main():
    # 1. Download data
    if not os.path.exists(RAW_DATA_PATH):
        print('Downloading data...')
        data = download_data(TICKER, START_DATE, END_DATE)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        data.to_csv(RAW_DATA_PATH)
    else:
        print('Loading raw data...')
        # Handle the multi-row header from previous download
        data = pd.read_csv(RAW_DATA_PATH, index_col=0, header=[0, 1, 2])
        # Simplify columns if they are multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    # 2. Preprocess data
    print('Preprocessing data...')
    # Ensure all data is numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    data = preprocess_data(data)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    data.to_csv(PROCESSED_DATA_PATH)

    # 3. Normalize and create sequences
    print('Normalizing and creating sequences...')
    data_norm, scaler = normalize_data(data)
    save_scaler(scaler, SCALER_PATH)

    # Sequence based data for DL models
    X, y = lag_data(data_norm)

    # 4. Split data
    print('Splitting data...')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Build and train model (Example: GRU with Attention)
    print('Building and training GRU model...')
    gru_model = GRUModel(use_attention=True, seq_length=X_train.shape[1], n_features=X_train.shape[2])
    gru_model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    save_model(gru_model, MODEL_PATH)

    # 6. Evaluate model
    print('Evaluating model...')
    metrics = {'MSE': MSE, 'MAPE': MAPE, 'RMSE': RMSE, 'SMAPE': SMAPE, 'R2': R2, 'MASE': MASE, 'RMSSE': RMSSE, 'MDA': MDA}
    results = evaluate(gru_model, metrics, X_test, y_test)
    print('Evaluation Results:', results)

    # 7. VAR Model example
    print('Training VAR model...')
    train_size = int(len(data_norm) * 0.8)
    var_train = data_norm.iloc[:train_size]
    var_test = data_norm.iloc[train_size:]

    var_model = VARModel(lags=4)
    var_model.fit(var_train)

    # Predict next steps
    var_preds = var_model.predict(var_train.values, steps=len(var_test))

    # 8. Unscale and Plot (Example for GRU)
    print('Plotting results...')
    y_test_unscaled = unscale_data(y_test.flatten(), scaler)
    y_pred_unscaled = unscale_data(gru_model.predict(X_test).flatten(), scaler)

    print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
