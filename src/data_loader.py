import yfinance as yf
import pandas as pd
import numpy as np


class DataLoader:
    """
    A class for loading and preprocessing financial market data.

    This class fetches historical price data for specified stock tickers
    using the yfinance API, processes the data by calculating daily returns
    and volatility, and provides methods to combine returns data
    across multiple tickers.

    Args:
        start_date (str): The start date for historical data in 'YYYY-MM-DD' format
        end_date (str): The end date for historical data in 'YYYY-MM-DD' format
        tickers (list): List of stock ticker symbols to fetch data for
    """

    def __init__(self, start_date="2015-07-01", end_date="2025-07-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = ["TSLA", "BND", "SPY"]

    def fetch_data(self):
        """
        Fetch historical price data for the specified stock tickers.

        Return:
            pd.DataFrame: A DataFrame containing historical price data for 
            the tickers.
        """
        print("Calculating Basic Statistics for the closing price . . . . .  \n")
        # download the data from yfinance
        data = yf.download(self.tickers, self.start_date, self.end_date)
        # drop the rows with missing values
        data = data.dropna()

        return data

    def calculate(self, df):
        """
        Calculate daily returns and 30-day rolling volatility for the given
        DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing historical price data 
                                with a 'Close' column.

        Returns:
            tuple: A tuple containing:
                - daily_return (pd.Series): Daily percentage returns of the closing price.
                - Volatility (pd.Series): 30-day rolling standard deviation of daily returns.
        """
        # Calculate daily percentage returns for the 'Close' price
        daily_return = df["Close"].pct_change().dropna()
        # Compute 30-day rolling standard deviation (volatility) of those returns
        Volatility = daily_return.rolling(window=30).std()
        return daily_return, Volatility
