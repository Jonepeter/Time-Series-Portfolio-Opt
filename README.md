# Time Series Portfolio Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced portfolio optimization system leveraging time series forecasting for strategic asset allocation

## 📋 Overview

This project implements a comprehensive portfolio optimization system for **GMF Investments**, utilizing advanced time series forecasting techniques to optimize asset allocation across **TSLA**, **BND**, and **SPY**. The system combines traditional statistical methods (ARIMA) with deep learning approaches (LSTM) to predict market trends and maximize portfolio performance.

## ✨ Features

- 📊 **Comprehensive EDA** - In-depth exploratory data analysis with visualization
- 🔮 **Dual Forecasting Models** - ARIMA and LSTM implementations
- 📈 **Market Trend Prediction** - Future price movement forecasting
- 💼 **Portfolio Optimization** - Risk-adjusted return maximization
- 🔄 **Strategy Backtesting** - Historical performance validation
- 📱 **Interactive Notebooks** - Jupyter-based analysis workflow

## 🏗️ Project Structure

```
Time-Series-Portfolio-Opt/
├── 📁 src/                     # Core source code
│   ├── main.py                 # Main execution script
│   ├── data_loader.py          # Data fetching and preprocessing
│   ├── eda_analysis.py         # Exploratory data analysis
│   ├── forecasting_models.py   # ARIMA & LSTM models
│   ├── portfolio_optimizer.py  # Portfolio optimization logic
│   └── backtester.py          # Strategy backtesting
├── 📁 notebooks/               # Jupyter analysis notebooks
│   ├── 1_eda.ipynb            # Exploratory data analysis
│   ├── 2_forecasting.ipynb    # Time series forecasting
│   ├── 4_portfolio_optimization.ipynb
│   └── 5_backtesting.ipynb    # Strategy validation
├── 📁 data/                   # Raw and processed datasets
├── 📁 results/                # Generated plots and outputs
├── 📁 reports/                # Project documentation
├── 📁 tests/                  # Unit tests
└── 📄 requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Time-Series-Portfolio-Opt
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main application**

   ```bash
   python src/main.py
   ```

### Alternative: Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

## 📊 Usage

### Command Line Interface

```bash
# Run complete pipeline
python src/main.py

# Run specific components
python src/eda_analysis.py        # EDA only
python src/forecasting_models.py  # Forecasting only
python src/portfolio_optimizer.py # Optimization only
```

### Programmatic Usage

```python
from src.data_loader import DataLoader
from src.forecasting_models import LSTMForecaster
from src.portfolio_optimizer import PortfolioOptimizer

# Load data
loader = DataLoader(['TSLA', 'BND', 'SPY'])
data = loader.fetch_data()

# Generate forecasts
forecaster = LSTMForecaster()
predictions = forecaster.predict(data)

# Optimize portfolio
optimizer = PortfolioOptimizer()
weights = optimizer.optimize(predictions)
```

## 🛠️ Technologies

| Category | Technologies |
|----------|-------------|
| **Data Processing** | pandas, numpy, yfinance |
| **Machine Learning** | scikit-learn, TensorFlow, statsmodels |
| **Visualization** | matplotlib, seaborn |
| **Time Series** | pmdarima, scipy |
| **Development** | Python 3.8+, Jupyter |

## 📈 Methodology

1. **Data Collection** - Historical price data via Yahoo Finance API
2. **Exploratory Analysis** - Statistical analysis and visualization
3. **Feature Engineering** - Technical indicators and lag features
4. **Model Training** - ARIMA and LSTM model development
5. **Forecasting** - Multi-step ahead predictions
6. **Optimization** - Modern Portfolio Theory implementation
7. **Backtesting** - Historical performance validation

## 📊 Results

The system generates comprehensive outputs including:
- 📈 Price forecasting charts
- 💹 Volatility analysis plots
- 🎯 Optimized portfolio weights
- 📋 Performance metrics and reports

## 🧪 Testing

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src

# Run specific test
python -m pytest tests/test_data_loader.py

# Run automated test suite
python run_tests.py

# Performance monitoring
python scripts/performance_monitor.py
```

## 🔧 Development

```bash
# Code formatting
python -m black src/ tests/
python -m isort src/ tests/

# Linting
python -m flake8 src/ tests/

# Environment setup
cp .env.example .env
# Edit .env with your settings
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

**GMF Investments** - Portfolio Optimization Team

---

⭐ **Star this repository if you found it helpful!**