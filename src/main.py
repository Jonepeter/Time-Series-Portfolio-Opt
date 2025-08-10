from data_loader import DataLoader
from eda_analysis import EDAAnalysis
from forecasting_models import ForecastingModels
from portfolio_optimizer import PortfolioOptimizer
from backtester import Backtester
import pandas as pd
import numpy as np

def main():
    print("=== GMF Investments Portfolio Optimization System ===\n")
    
    # Task 1: Data Loading and EDA
    print("Task 1: Loading and preprocessing data...")
    loader = DataLoader()
    raw_data = loader.fetch_data()
    processed_data = loader.preprocess_data(raw_data)
    returns_data = loader.get_combined_returns(processed_data)
    
    print("Data loaded successfully!")
    print(f"Data shape: {returns_data.shape}")
    print(f"Date range: {returns_data.index[0]} to {returns_data.index[-1]}\n")
    
    # EDA Analysis
    print("Performing EDA analysis...")
    eda = EDAAnalysis(processed_data)
    stats = eda.basic_statistics()
    print("Basic Statistics:")
    print(stats)
    
    stationarity = eda.stationarity_test()
    print("\nStationarity Test Results:")
    print(stationarity)
    
    risk_metrics = eda.calculate_risk_metrics('TSLA')
    print(f"\nTSLA Risk Metrics: {risk_metrics}\n")
    
    # Task 2: Forecasting Models
    print("Task 2: Building forecasting models...")
    forecaster = ForecastingModels(processed_data, 'TSLA')
    model_results = forecaster.evaluate_models()
    print(f"ARIMA Model - MAE: {model_results['arima_mae']:.4f}, RMSE: {model_results['arima_rmse']:.4f}\n")
    
    # Task 3: Future Forecasting
    print("Task 3: Generating future forecasts...")
    arima_model = model_results['arima_model']
    future_forecast = arima_model.forecast(steps=252)  # 1 year forecast
    print(f"Generated {len(future_forecast)} day forecast\n")
    
    # Task 4: Portfolio Optimization
    print("Task 4: Optimizing portfolio...")
    optimizer = PortfolioOptimizer(returns_data)
    
    max_sharpe_weights = optimizer.optimize_portfolio('sharpe')
    min_vol_weights = optimizer.optimize_portfolio('volatility')
    
    print("Optimal Portfolio Weights:")
    assets = ['TSLA', 'BND', 'SPY']
    for i, asset in enumerate(assets):
        print(f"{asset}: {max_sharpe_weights[i]:.3f}")
    
    max_sharpe_return, max_sharpe_vol = optimizer.portfolio_performance(max_sharpe_weights)
    sharpe_ratio = (max_sharpe_return - 0.02) / max_sharpe_vol
    
    print(f"\nPortfolio Metrics:")
    print(f"Expected Annual Return: {max_sharpe_return:.3f}")
    print(f"Annual Volatility: {max_sharpe_vol:.3f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}\n")
    
    # Task 5: Backtesting
    print("Task 5: Backtesting strategy...")
    backtester = Backtester(returns_data, max_sharpe_weights)
    performance_comparison = backtester.plot_backtest_results()
    
    print("Performance Comparison:")
    print(performance_comparison)
    
    print("\n=== Analysis Complete ===")
    
    # Save results
    results = {
        'optimal_weights': dict(zip(assets, max_sharpe_weights)),
        'portfolio_metrics': {
            'expected_return': max_sharpe_return,
            'volatility': max_sharpe_vol,
            'sharpe_ratio': sharpe_ratio
        },
        'forecast_summary': {
            'forecast_mean': future_forecast.mean(),
            'forecast_std': future_forecast.std()
        }
    }
    
    pd.DataFrame([results]).to_json('../results/portfolio_results.json', indent=2)
    print("Results saved to results/portfolio_results.json")

if __name__ == "__main__":
    main()