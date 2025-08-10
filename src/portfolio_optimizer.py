import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, returns_data):
        self.returns = returns_data
        self.mean_returns = returns_data.mean() * 252
        self.cov_matrix = returns_data.cov() * 252
        
    def portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_std
    
    def negative_sharpe(self, weights, risk_free_rate=0.02):
        p_return, p_std = self.portfolio_performance(weights)
        return -(p_return - risk_free_rate) / p_std
    
    def minimize_volatility(self, weights):
        return self.portfolio_performance(weights)[1]
    
    def optimize_portfolio(self, target='sharpe'):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        
        if target == 'sharpe':
            result = minimize(self.negative_sharpe, initial_guess, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            result = minimize(self.minimize_volatility, initial_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def efficient_frontier(self, num_portfolios=100):
        results = np.zeros((3, num_portfolios))
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), num_portfolios)
        
        for i, target in enumerate(target_returns):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                          {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target})
            bounds = tuple((0, 1) for _ in range(len(self.mean_returns)))
            initial_guess = len(self.mean_returns) * [1. / len(self.mean_returns)]
            
            result = minimize(self.minimize_volatility, initial_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                results[0, i] = target
                results[1, i] = result.fun
                results[2, i] = (target - 0.02) / result.fun
        
        return results
    
    def plot_efficient_frontier(self):
        frontier = self.efficient_frontier()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(frontier[1], frontier[0], c=frontier[2], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        
        # Mark optimal portfolios
        max_sharpe_weights = self.optimize_portfolio('sharpe')
        min_vol_weights = self.optimize_portfolio('volatility')
        
        max_sharpe_return, max_sharpe_vol = self.portfolio_performance(max_sharpe_weights)
        min_vol_return, min_vol_vol = self.portfolio_performance(min_vol_weights)
        
        plt.scatter(max_sharpe_vol, max_sharpe_return, marker='*', color='red', s=500, label='Max Sharpe')
        plt.scatter(min_vol_vol, min_vol_return, marker='*', color='green', s=500, label='Min Volatility')
        plt.legend()
        plt.show()
        
        return max_sharpe_weights, min_vol_weights