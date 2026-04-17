import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from .base_model import BaseModel

class VARModel(BaseModel):
    def __init__(self, lags=4):
        super().__init__()
        self.lags = lags
        self.model = None
        self.results = None

    def fit(self, X, y=None):
        """
        Fit the VAR model.
        X: pd.DataFrame or np.array - training data
        y: ignored for VAR but kept for API consistency
        """
        self.model = VAR(X)
        self.results = self.model.fit(self.lags)
        return self.results

    def predict(self, X, steps=1):
        """
        Predict using the VAR model.
        X: the data used to forecast (the last 'lags' observations)
        steps: number of steps to forecast
        """
        if self.results is None:
            raise ValueError("Model must be fitted before predicting.")

        # statsmodels forecast expects (lags, n_features)
        # if X has more than lags, take the last ones
        if len(X) > self.lags:
            forecast_input = X[-self.lags:]
        else:
            forecast_input = X

        forecast = self.results.forecast(forecast_input, steps=steps)
        return forecast

    def save(self, path):
        if self.results:
            self.results.save(path)

    def load(self, path):
        from statsmodels.tsa.vector_ar.var_model import VARResults
        # statsmodels loading is usually via pickle or its own load method
        pass
