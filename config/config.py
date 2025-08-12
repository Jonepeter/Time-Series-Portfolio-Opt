"""
Configuration settings for Time Series Portfolio Optimization
"""
import os
from datetime import datetime

class Config:
    """Main configuration class"""
    
    # Data Configuration
    START_DATE = "2015-07-01"
    END_DATE = "2024-12-31"
    TICKERS = ["TSLA", "BND", "SPY"]
    
    # Model Configuration
    RANDOM_SEED = 42
    VOLATILITY_WINDOW = 30
    FORECAST_HORIZON = 252  # 1 year
    
    # Portfolio Optimization
    RISK_FREE_RATE = 0.02
    OPTIMIZATION_METHODS = ['sharpe', 'volatility']
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Model Parameters
    ARIMA_MAX_P = 5
    ARIMA_MAX_D = 2
    ARIMA_MAX_Q = 5
    
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    LSTM_SEQUENCE_LENGTH = 60
    
    @classmethod
    def get_data_path(cls, filename):
        """Get full path for data file"""
        return os.path.join(cls.DATA_DIR, filename)
    
    @classmethod
    def get_results_path(cls, filename):
        """Get full path for results file"""
        return os.path.join(cls.RESULTS_DIR, filename)
    
    @classmethod
    def ensure_directories(cls):
        """Create directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.RESULTS_DIR, cls.REPORTS_DIR]:
            os.makedirs(directory, exist_ok=True)