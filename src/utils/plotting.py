import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(y_true, y_pred, title="Model Predictions"):
    """
    Plot actual vs predicted values.
    y_true: actual values (numpy array or series)
    y_pred: predicted values (numpy array or series)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual Price", alpha=0.7)
    plt.plot(y_pred, label="Predicted Price", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_technical_indicators(data, indicators):
    """
    Plot selected technical indicators.
    data: pd.DataFrame containing the indicators
    indicators: list of column names to plot
    """
    plt.figure(figsize=(14, 10))
    for i, indicator in enumerate(indicators):
        plt.subplot(len(indicators), 1, i+1)
        plt.plot(data[indicator], label=indicator)
        plt.title(f"Technical Indicator: {indicator}")
        plt.legend()
    plt.tight_layout()
    plt.show()
