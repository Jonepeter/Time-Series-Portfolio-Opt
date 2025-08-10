import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, returns_data, weights, benchmark_weights=None):
        self.returns = returns_data
        self.weights = weights
        self.benchmark_weights = benchmark_weights or [0, 0.4, 0.6]  # BND: 40%, SPY: 60%
        
    def calculate_portfolio_returns(self, weights):
        return (self.returns * weights).sum(axis=1)
    
    def backtest_strategy(self, start_date=None):
        if start_date:
            backtest_returns = self.returns[start_date:]
        else:
            backtest_returns = self.returns[-252:]  # Last year
            
        strategy_returns = (backtest_returns * self.weights).sum(axis=1)
        benchmark_returns = (backtest_returns * self.benchmark_weights).sum(axis=1)
        
        strategy_cumulative = (1 + strategy_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        return {
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'strategy_cumulative': strategy_cumulative,
            'benchmark_cumulative': benchmark_cumulative
        }
    
    def calculate_metrics(self, returns):
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def plot_backtest_results(self):
        results = self.backtest_strategy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(results['strategy_cumulative'].index, results['strategy_cumulative'], 
                label='Strategy', linewidth=2)
        plt.plot(results['benchmark_cumulative'].index, results['benchmark_cumulative'], 
                label='Benchmark (60/40)', linewidth=2)
        plt.title('Strategy vs Benchmark Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        strategy_metrics = self.calculate_metrics(results['strategy_returns'])
        benchmark_metrics = self.calculate_metrics(results['benchmark_returns'])
        
        comparison = pd.DataFrame({
            'Strategy': strategy_metrics,
            'Benchmark': benchmark_metrics
        })
        
        return comparison