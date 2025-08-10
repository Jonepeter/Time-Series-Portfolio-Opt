import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    
    def __init__(self, data, target_asset='TSLA'):
        self.data = data[target_asset]['Close']
        self.target_asset = target_asset
        self.train_size = int(len(self.data) * 0.8)
        
    def build_arima_model(self):
        train_data = self.data[:self.train_size]
        
        # Manual parameter selection (replaces auto_arima)
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p, d, q in product(range(3), range(2), range(3)):
            try:
                model = ARIMA(train_data, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_order = (p, d, q)
            except:
                continue
                
        model = ARIMA(train_data, order=best_order)
        return model.fit()
    
    def build_lstm_model(self, sequence_length=60):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.data.values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        return model, scaler
    
    def evaluate_models(self):
        arima_model = self.build_arima_model()
        test_data = self.data[self.train_size:]
        arima_forecast = arima_model.forecast(steps=len(test_data))
        
        arima_mae = mean_absolute_error(test_data, arima_forecast)
        arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
        
        return {
            'arima_model': arima_model,
            'arima_mae': arima_mae,
            'arima_rmse': arima_rmse
        }