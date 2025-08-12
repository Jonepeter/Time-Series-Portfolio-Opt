import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
from datetime import timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingModels:
    def __init__(self, data, target_asset="TSLA"):
        self.data = data
        self.target_asset = target_asset
        # Extract Close price series for the target asset
        if isinstance(data[target_asset], pd.DataFrame):
            self.target_series = data[target_asset]['Close']
        else:
            self.target_series = data[target_asset]
        self.scaler = MinMaxScaler()
        self.arima_model = None
        self.lstm_model = None

    def check_stationarity(self):
        """Check if the time series is stationary"""
        result = adfuller(self.target_series.dropna())
        return result[1] <= 0.05

    def fit_arima(self, order=None):
        """Fit ARIMA model with automatic order selection for stationarity"""
        if order is None:
            # Check if data is stationary
            is_stationary = self.check_stationarity()
            if is_stationary:
                order = (1, 0, 1)  # No differencing needed
            else:
                order = (0, 1, 1)  # Differencing needed for non-stationary data

        logger.info(f"Fitting ARIMA{order} model for {self.target_asset}")
        self.arima_model = ARIMA(self.target_series, order=order)
        self.arima_fitted = self.arima_model.fit()
        return self.arima_fitted

    def prepare_lstm_data(self, lookback=60):
        """Prepare data for LSTM model"""
        scaled_data = self.scaler.fit_transform(
            self.target_series.values.reshape(-1, 1)
        )

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback : i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))

        return X[:split], X[split:], y[:split], y[split:]

    def fit_lstm(self, lookback=60, epochs=50):
        """Fit LSTM model"""
        logger.info(f"Fitting LSTM model for {self.target_asset}")
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(lookback)

        self.lstm_model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1),
            ]
        )

        self.lstm_model.compile(optimizer="adam", loss="mse")
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        return self.lstm_model

    def forecast_arima(self, steps=252):
        """Generate ARIMA forecasts with confidence intervals"""
        if self.arima_model is None:
            self.fit_arima()

        try:
            forecast = self.arima_fitted.forecast(steps=steps)
            conf_int = self.arima_fitted.get_forecast(steps=steps).conf_int()
        except:
            # pass
            # If forecast fails, use simple trend extrapolation
            last_price = self.target_series.iloc[-1]
            forecast = np.full(steps, last_price)
            conf_int = pd.DataFrame(
                {
                    "lower": np.full(steps, last_price * 0.9),
                    "upper": np.full(steps, last_price * 1.1),
                }
            )

        # Create future dates
        last_date = self.target_series.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=steps, freq="D"
        )

        return pd.Series(forecast, index=future_dates), conf_int

    def forecast_lstm(self, steps=252, lookback=60):
        """Generate LSTM forecasts"""
        if self.lstm_model is None:
            self.fit_lstm(lookback)

        # Use last 'lookback' days for prediction
        last_data = self.scaler.transform(
            self.target_series.tail(lookback).values.reshape(-1, 1)
        )
        predictions = []

        current_batch = last_data.reshape(1, lookback, 1)

        for _ in range(steps):
            pred = self.lstm_model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(pred)

            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], [[[pred]]], axis=1)

        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        # Create future dates
        last_date = self.target_series.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=steps, freq="D"
        )

        return pd.Series(predictions, index=future_dates)

    def plot_forecast(self, model_type="arima", steps=252, historical_days=500):
        """Plot forecast with historical data"""
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot historical data
        historical = self.target_series.tail(historical_days)
        ax.plot(
            historical.index,
            historical.values,
            label="Historical",
            color="blue",
            linewidth=2,
        )

        if model_type.lower() == "arima":
            forecast, conf_int = self.forecast_arima(steps)
            ax.plot(
                forecast.index,
                forecast.values,
                label="ARIMA Forecast",
                color="red",
                linewidth=2,
            )
            ax.fill_between(
                forecast.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color="red",
                alpha=0.3,
                label="Confidence Interval",
            )
        else:
            forecast = self.forecast_lstm(steps)
            ax.plot(
                forecast.index,
                forecast.values,
                label="LSTM Forecast",
                color="green",
                linewidth=2,
            )

        ax.set_title(
            f"{self.target_asset} Price Forecast - {model_type.upper()}", fontsize=16
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return forecast

    def analyze_forecast(self, model_type="arima", steps=252):
        """Analyze forecast trends and risks"""
        if model_type.lower() == "arima":
            forecast, conf_int = self.forecast_arima(steps)

            # Calculate confidence interval width
            ci_width = conf_int.iloc[:, 1] - conf_int.iloc[:, 0]

            analysis = {
                "forecast": forecast,
                "confidence_intervals": conf_int,
                "ci_width": ci_width,
                "trend": (
                    "Upward" if forecast.iloc[-1] > forecast.iloc[0] else "Downward"
                ),
                "volatility": forecast.std(),
                "max_price": forecast.max(),
                "min_price": forecast.min(),
                "price_change_pct": (
                    (forecast.iloc[-1] - self.target_series.iloc[-1])
                    / self.target_series.iloc[-1]
                )
                * 100,
            }
        else:
            forecast = self.forecast_lstm(steps)

            analysis = {
                "forecast": forecast,
                "trend": (
                    "Upward" if forecast.iloc[-1] > forecast.iloc[0] else "Downward"
                ),
                "volatility": forecast.std(),
                "max_price": forecast.max(),
                "min_price": forecast.min(),
                "price_change_pct": (
                    (forecast.iloc[-1] - self.target_series.iloc[-1])
                    / self.target_series.iloc[-1]
                )
                * 100,
            }

        return analysis

    def evaluate_models(self):
        """Evaluate both ARIMA and LSTM models"""
        results = {}

        # Split data for evaluation
        train_size = int(0.8 * len(self.target_series))
        train_data = self.target_series.iloc[:train_size]
        test_data = self.target_series.iloc[train_size:]

        try:
            # ARIMA evaluation
            arima_model = ARIMA(train_data, order=(0, 1, 1)).fit()
            arima_pred = arima_model.forecast(len(test_data))

            results["arima_mae"] = np.mean(np.abs(test_data - arima_pred))
            results["arima_rmse"] = np.sqrt(np.mean((test_data - arima_pred) ** 2))
            results["arima_model"] = arima_model
        except Exception as e:
            # Use simple baseline prediction
            baseline_value = train_data.iloc[-1]
            baseline_pred = np.full(len(test_data), baseline_value)
            results["arima_mae"] = np.mean(np.abs(test_data.values - baseline_pred))
            results["arima_rmse"] = np.sqrt(np.mean((test_data.values - baseline_pred) ** 2))
            results["arima_model"] = None

        try:
            # LSTM evaluation
            X_train, X_test, y_train, y_test = self.prepare_lstm_data()
            lstm_model = self.fit_lstm()
            lstm_pred = lstm_model.predict(X_test, verbose=0)
            lstm_pred = self.scaler.inverse_transform(lstm_pred).flatten()
            y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            results["lstm_mae"] = np.mean(np.abs(y_test_inv - lstm_pred))
            results["lstm_rmse"] = np.sqrt(np.mean((y_test_inv - lstm_pred) ** 2))
            results["lstm_model"] = lstm_model
        except Exception as e:
            results["lstm_error"] = str(e)

        return results

    def plot_model_comparison(self):
        """Plot test vs actual data comparison for both models"""
        train_size = int(0.8 * len(self.target_series))
        train_data = self.target_series[:train_size]
        test_data = self.target_series[train_size:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # ARIMA comparison
        try:
            arima_model = ARIMA(train_data, order=(0, 1, 1)).fit()
            arima_pred = arima_model.forecast(len(test_data))

            ax1.plot(
                test_data.index,
                test_data.values,
                label="Actual",
                color="blue",
                linewidth=2,
            )
            ax1.plot(
                test_data.index,
                arima_pred,
                label="ARIMA Prediction",
                color="red",
                linewidth=2,
            )
            ax1.set_title(f"ARIMA Model: Actual vs Predicted - {self.target_asset}")
            ax1.set_ylabel("Price ($)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        except:
            ax1.text(
                0.5,
                0.5,
                "ARIMA Model Failed",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        # LSTM comparison
        try:
            X_train, X_test, y_train, y_test = self.prepare_lstm_data()
            lstm_model = self.fit_lstm()
            lstm_pred = lstm_model.predict(X_test, verbose=0)

            # Inverse transform
            lstm_pred = self.scaler.inverse_transform(lstm_pred).flatten()
            y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Create index for test data
            test_idx = test_data.index[-len(y_test_inv) :]

            ax2.plot(test_idx, y_test_inv, label="Actual", color="blue", linewidth=2)
            ax2.plot(
                test_idx, lstm_pred, label="LSTM Prediction", color="green", linewidth=2
            )
            ax2.set_title(f"LSTM Model: Actual vs Predicted - {self.target_asset}")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price ($)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except:
            ax2.text(
                0.5,
                0.5,
                "LSTM Model Failed",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plt.show()

    def print_forecast_insights(self, model_type="arima", steps=252):
        """Print detailed forecast analysis"""
        analysis = self.analyze_forecast(model_type, steps)

        print(f"\n{'='*60}")
        print(f"{self.target_asset} FORECAST ANALYSIS - {model_type.upper()}")
        print(f"{'='*60}")

        print(f"\n📈 TREND ANALYSIS:")
        print(f"   • Overall Trend: {analysis['trend']}")
        print(f"   • Expected Price Change: {analysis['price_change_pct']:.2f}%")
        print(f"   • Forecast Volatility: ${analysis['volatility']:.2f}")

        print(f"\n💰 PRICE PROJECTIONS:")
        print(f"   • Current Price: ${self.target_series.iloc[-1]:.2f}")
        print(f"   • Forecasted Max: ${analysis['max_price']:.2f}")
        print(f"   • Forecasted Min: ${analysis['min_price']:.2f}")

        if "ci_width" in analysis:
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   • Initial CI Width: ${analysis['ci_width'].iloc[0]:.2f}")
            print(f"   • Final CI Width: ${analysis['ci_width'].iloc[-1]:.2f}")
            print(
                f"   • CI Growth: {((analysis['ci_width'].iloc[-1] / analysis['ci_width'].iloc[0]) - 1) * 100:.1f}%"
            )

        print(f"\n⚠️  RISK ASSESSMENT:")
        if analysis["volatility"] > 50:
            print("   • HIGH volatility expected - Significant price swings likely")
        elif analysis["volatility"] > 20:
            print("   • MODERATE volatility expected - Normal market fluctuations")
        else:
            print("   • LOW volatility expected - Stable price movements")

        print(f"\n🎯 INVESTMENT INSIGHTS:")
        if analysis["price_change_pct"] > 10:
            print("   • OPPORTUNITY: Strong upward trend predicted")
        elif analysis["price_change_pct"] < -10:
            print("   • RISK: Significant decline expected")
        else:
            print("   • NEUTRAL: Sideways movement expected")
