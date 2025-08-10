import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from data_loader import DataLoader
from statsmodels.tsa.stattools import adfuller

# Configure logging
logger = logging.getLogger(__name__)

class EDAAnalysis:
    """
    A class for performing Exploratory Data Analysis on financial market data.
    
    This class provides methods for statistical analysis, visualization,
    and financial metric calculations including VaR and Sharpe Ratio.
    
    Attributes:
        loader (DataLoader): Instance of DataLoader for data fetching
        data (pd.DataFrame): Raw financial data
    """
    
    def __init__(self):
        """
        Initialize EDAAnalysis with data loader and fetch data.
        
        Raises:
            ConnectionError: If data fetching fails
        """
        try:
            self.loader = DataLoader()
            self.data = self.loader.fetch_data()
            logger.info("EDAAnalysis initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EDAAnalysis: {e}")
            raise
        
    def basic_statistics(self):
        """
        Calculate basic statistics for the daily returns of the stock data.
        
        Returns:
            pd.DataFrame: A DataFrame containing comprehensive statistics including
                         mean, std, variance, skewness, and kurtosis
                         
        Raises:
            ValueError: If data is insufficient for statistical calculations
        """
        try:
            daily_return, _ = self.loader.calculate(self.data)
            
            if daily_return.empty:
                raise ValueError("No return data available for statistical analysis")
            
            stats = daily_return.describe().T[['count','mean', 'std', 'min', 'max']]
            
            # Add additional statistical descriptions   
            stats['Variance'] = daily_return.var() # Variance used to measure the dispersion of returns
            stats['Skewness'] = daily_return.skew() # Skewness used to measure the asymmetry of the distribution
            stats['Kurtosis'] = daily_return.kurtosis() # Kurtosis used to measure the "tailedness" of the distribution
            
            logger.info("Basic statistics calculated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            raise
    
    def plot_closing_prices(self):
        """
        Visualize the closing price over time for each ticker.
        
        Raises:
            ValueError: If closing price data is not available
            KeyError: If 'Close' column is missing from data
        """
        try:
            if 'Close' not in self.data.columns.levels[0]:
                raise KeyError("'Close' price data not found")
            
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
            
            logger.info("Closing prices plot generated successfully")
            
        except Exception as e:
            logger.error(f"Error plotting closing prices: {e}")
            raise
    
    def plot_daily_return(self):
        """
        Calculate and plot the daily percentage change (returns) for each ticker.
        
        Returns:
            pd.DataFrame: Descriptive statistics of daily returns
            
        Raises:
            KeyError: If 'Close' price data is not available
            ValueError: If insufficient data for return calculations
        """
        try:
            if 'Close' not in self.data.columns.levels[0]:
                raise KeyError("'Close' price data not found")
            
            daily_pct_change = self.data['Close'].pct_change().dropna()
            
            if daily_pct_change.empty:
                raise ValueError("No return data available for plotting")
            
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
            
            logger.info("Daily returns plot generated successfully")
            return daily_pct_change.describe().T
            
        except Exception as e:
            logger.error(f"Error plotting daily returns: {e}")
            raise
    
    def plot_volatility(self, window=30):
        """
        Calculate rolling volatility and plot for each ticker.
        
        Args:
            window (int): Rolling window size for volatility calculation (default: 30)
            
        Returns:
            pd.DataFrame: Rolling volatility data
            
        Raises:
            ValueError: If window size is invalid or insufficient data
            KeyError: If 'Close' price data is not available
        """
        try:
            if window <= 0:
                raise ValueError("Window size must be positive")
            
            if 'Close' not in self.data.columns.levels[0]:
                raise KeyError("'Close' price data not found")
            
            daily_returns = self.data['Close'].pct_change().dropna()
            
            if len(daily_returns) < window:
                raise ValueError(f"Insufficient data for {window}-day rolling volatility")
            
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
            
            logger.info(f"Volatility plot generated for {window}-day window")
            return daily_returns.rolling(window).std().dropna()
            
        except Exception as e:
            logger.error(f"Error plotting volatility: {e}")
            raise

    def test_stationarity(self, column='Close', alpha=0.05):
        """
        Perform Augmented Dickey-Fuller test for each ticker's time series.
        
        Args:
            column (str): Column name to test for stationarity (default: 'Close')
            alpha (float): Significance level for the test (default: 0.05)
            
        Returns:
            pd.DataFrame: DataFrame with ADF statistics and stationarity results
            
        Raises:
            KeyError: If specified column is not found
            ValueError: If insufficient data for stationarity test
        """
        try:
            if column not in self.data.columns.levels[0]:
                raise KeyError(f"Column '{column}' not found in data")
            
            results = {}
            for ticker in self.data[column].columns:
                try:
                    series = self.data[column][ticker].dropna()
                    
                    if len(series) < 10:
                        logger.warning(f"Insufficient data for {ticker} stationarity test")
                        continue
                    
                    result = adfuller(series)
                    p_value = result[1]
                    stationary = "Stationary" if p_value < alpha else "Non-Stationary"
                    
                    results[ticker] = {
                        'adf_stat': result[0],
                        'p_value': p_value,
                        'used_lag': result[2],
                        'n_obs': result[3],
                        'stationarity': stationary
                    }
                    
                except Exception as e:
                    logger.warning(f"Error testing stationarity for {ticker}: {e}")
                    continue
            
            if not results:
                raise ValueError("No valid stationarity test results")
            
            logger.info("Stationarity tests completed successfully")
            return pd.DataFrame(results).T
            
        except Exception as e:
            logger.error(f"Error in stationarity testing: {e}")
            raise
        
    def calculate_risk_metrics(self, ticker='TSLA'):
        """
        Calculate Value at Risk (VaR) and Sharpe Ratio for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol to calculate metrics for (default: 'TSLA')
            
        Returns:
            dict: Dictionary containing VaR_95 and Sharpe_Ratio
            
        Raises:
            KeyError: If ticker is not found in the data
            ValueError: If insufficient data for calculations
        """
        try:
            daily_return, _ = self.loader.calculate(self.data)
            
            if ticker not in daily_return.columns:
                raise KeyError(f"Ticker '{ticker}' not found in return data")
            
            returns = daily_return[ticker].dropna()
            
            if len(returns) < 30:
                raise ValueError(f"Insufficient data for {ticker} risk calculations")
            
            # Calculate 95% Value at Risk (5th percentile)
            var_95 = np.percentile(returns, 5)
            
            # Calculate annualized Sharpe Ratio (assuming risk-free rate = 0)
            if returns.std() == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            
            logger.info(f"Risk metrics calculated for {ticker}")
            return {'VaR_95': var_95, 'Sharpe_Ratio': sharpe_ratio}
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {ticker}: {e}")
            raise