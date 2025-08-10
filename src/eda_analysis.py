import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from statsmodels.tsa.stattools import adfuller

class EDAAnalysis:
    def __init__(self):
        # self.data = data
        self.loader = DataLoader()
        self.data = self.loader.fetch_data()
        
    def basic_statistics(self):
        """
        Calculate basic statistics for the daily returns of the stock data.
        
        Returns:
            pd.DataFrame: A DataFrame containing mean, standard deviation, and other statistics.
        """
        # print("Calculating Basic Statistics for the Closing price . . . . .  ")
        daily_return, _ = self.loader.calculate(self.data)
        stats = daily_return.describe().T[['count','mean', 'std', 'min', 'max']]
        # Add additional statistical descriptions   
        stats['Variance'] = daily_return.var() # Variance used to measure the dispersion of returns
        stats['Skewness'] = daily_return.skew() # Skewness used to measure the asymmetry of the distribution
        stats['Kurtosis'] = daily_return.kurtosis() # Kurtosis used to measure the "tailedness" of the distribution
        return stats
    
    def plot_closing_prices(self):
        """
        Visualize the closing price over time for each ticker.
        """
        plt.figure(figsize=(12, 6))
        for ticker in self.data['Close'].columns:
            plt.plot(self.data['Close'].index, self.data['Close'][ticker], label=ticker)
        plt.title('Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_daily_return(self):
        """
        Calculate and plot the daily percentage change (returns) for each ticker.
        """
        daily_pct_change = self.data['Close'].pct_change().dropna()
        plt.figure(figsize=(12, 6))
        for ticker in daily_pct_change.columns:
            plt.plot(daily_pct_change.index, daily_pct_change[ticker], label=ticker)
        plt.title('Daily Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%) Change')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return daily_pct_change.describe().T
    
    def plot_volatility(self, window=30):
        """
        Calculate rolling volatility (standard deviation of daily returns) and plot for each ticker.
        """
        daily_returns = self.data['Close'].pct_change().dropna()
        plt.figure(figsize=(12, 6))
        for ticker in daily_returns.columns:
            rolling_vol = daily_returns[ticker].rolling(window).std()
            plt.plot(rolling_vol.index, rolling_vol, label=f"{ticker} ({window}d)")
        plt.title(f'Rolling {window}-Day Volatility (Std of Daily Returns)')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return daily_returns.rolling(window).std().dropna()

    def test_stationarity(self, column='Close', alpha=0.05):
        """
        Perform Augmented Dickey-Fuller test for each ticker's time series in the given column.
        Returns a DataFrame with ADF statistic, p-value, and stationarity conclusion for each ticker.
        """
        results = {}
        for ticker in self.data[column].columns:
            series = self.data[column][ticker].dropna()
            result = adfuller(series)
            p_value = result[1]
            stationary = "Stationary" if p_value < alpha else "Non-Stationary"
            results[ticker] = {
                'adf_stat': result[0],
                'p_value': p_value,
                'used_lag': result[2],
                'n_obs': result[3],
                # 'critical_values': result[4],
                'stationarity': stationary
            }
        return pd.DataFrame(results).T
        
    def calculate_risk_metrics(self, ticker='TSLA'):
        # df = self.data[ticker]
        daily_return, _ = self.loader.calculate(self.data)
        returns = daily_return['TSLA'].dropna()
        var_95 = np.percentile(returns, 5)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        return {'VaR_95': var_95, 'Sharpe_Ratio': sharpe_ratio}