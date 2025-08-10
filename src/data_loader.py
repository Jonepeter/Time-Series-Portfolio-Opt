import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading and preprocessing financial market data.

    This class fetches historical price data for specified stock tickers
    using the yfinance API, processes the data by calculating daily returns
    and volatility, and provides methods to combine returns data
    across multiple tickers.

    Attributes:
        start_date (str): The start date for historical data in 'YYYY-MM-DD' format
        end_date (str): The end date for historical data in 'YYYY-MM-DD' format
        tickers (list): List of stock ticker symbols to fetch data for
    """

    def __init__(self, start_date="2015-07-01", end_date="2024-12-31"):
        """
        Initialize the DataLoader with date range and tickers.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Raises:
            ValueError: If date format is invalid or start_date >= end_date
        """
        try:
            # Validate date formats
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
                
            self.start_date = start_date
            self.end_date = end_date
            self.tickers = ["TSLA", "BND", "SPY"]
            
            logger.info(f"DataLoader initialized with date range: {start_date} to {end_date}")
            
        except ValueError as e:
            logger.error(f"Invalid date format or range: {e}")
            raise

    def fetch_data(self):
        """
        Fetch historical price data for the specified stock tickers.

        Returns:
            pd.DataFrame: A DataFrame containing historical price data for the tickers
            
        Raises:
            ConnectionError: If unable to connect to Yahoo Finance API
            ValueError: If no data is returned for the specified tickers
        """
        try:
            logger.info("Fetching data from Yahoo Finance...")
            
            # Download data with error handling
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
            
            if data.empty:
                raise ValueError("No data returned for the specified tickers and date range")
            
            # Check for missing data
            initial_rows = len(data)
            data = data.dropna()
            final_rows = len(data)
            
            if final_rows == 0:
                raise ValueError("All data contains missing values")
            
            if initial_rows != final_rows:
                logger.warning(f"Dropped {initial_rows - final_rows} rows with missing values")
            
            logger.info(f"Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise ConnectionError(f"Failed to fetch data from Yahoo Finance: {e}")

    def calculate(self, df):
        """
        Calculate daily returns and 30-day rolling volatility for the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing historical price data with a 'Close' column

        Returns:
            tuple: A tuple containing:
                - daily_return (pd.Series): Daily percentage returns of the closing price
                - volatility (pd.Series): 30-day rolling standard deviation of daily returns
                
        Raises:
            KeyError: If 'Close' column is not found in the DataFrame
            ValueError: If DataFrame is empty or contains invalid data
        """
        try:
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            if "Close" not in df.columns:
                raise KeyError("'Close' column not found in DataFrame")
            
            # Calculate daily percentage returns for the 'Close' price
            daily_return = df["Close"].pct_change().dropna()
            
            if daily_return.empty:
                raise ValueError("Unable to calculate returns - insufficient data")
            
            # Compute 30-day rolling standard deviation (volatility) of those returns
            volatility = daily_return.rolling(window=30).std()
            
            logger.info("Successfully calculated daily returns and volatility")
            return daily_return, volatility
            
        except Exception as e:
            logger.error(f"Error calculating returns and volatility: {e}")
            raise
