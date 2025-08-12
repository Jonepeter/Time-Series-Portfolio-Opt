import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio_optimizer import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        self.returns_data = pd.DataFrame({
            'TSLA': np.random.normal(0.001, 0.03, 252),
            'BND': np.random.normal(0.0001, 0.005, 252),
            'SPY': np.random.normal(0.0005, 0.015, 252)
        }, index=dates)
        
        self.optimizer = PortfolioOptimizer(self.returns_data)

    def test_portfolio_performance(self):
        weights = np.array([0.4, 0.3, 0.3])
        portfolio_return, portfolio_vol = self.optimizer.portfolio_performance(weights)
        
        self.assertIsInstance(portfolio_return, float)
        self.assertIsInstance(portfolio_vol, float)
        self.assertGreater(portfolio_vol, 0)

    def test_optimize_portfolio_sharpe(self):
        weights = self.optimizer.optimize_portfolio('sharpe')
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_optimize_portfolio_volatility(self):
        weights = self.optimizer.optimize_portfolio('volatility')
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_negative_sharpe(self):
        weights = np.array([0.4, 0.3, 0.3])
        neg_sharpe = self.optimizer.negative_sharpe(weights)
        
        self.assertIsInstance(neg_sharpe, float)

    def test_efficient_frontier(self):
        frontier = self.optimizer.efficient_frontier(num_portfolios=10)
        
        self.assertEqual(frontier.shape, (3, 10))
        self.assertTrue(np.all(frontier[1] >= 0))  # Volatility should be positive


if __name__ == "__main__":
    unittest.main()