import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    """Portfolio backtesting system for Task 5 strategy validation.
    
    Simulates strategy performance over last year vs 60/40 benchmark.
    Supports monthly rebalancing and comprehensive performance analysis.
    
    Args:
        returns_data: DataFrame of daily returns for TSLA, BND, SPY
        weights: Array of optimal portfolio weights from Task 4
        benchmark_weights: Benchmark weights (default: 60% SPY, 40% BND)
    """

    def __init__(self, returns_data, weights, benchmark_weights=None):
        """Initialize backtester with returns data and portfolio weights."""
        self.returns = returns_data
        self.weights = np.array(weights)
        self.benchmark_weights = np.array(benchmark_weights or [0, 0.4, 0.6])  # TSLA: 0%, BND: 40%, SPY: 60%

    def calculate_portfolio_returns(self, weights):
        """Calculate portfolio returns given weights.

        Args:
            weights: Array of asset weights

        Returns:
            Series of portfolio returns
        """
        return (self.returns * weights).sum(axis=1)

    def backtest_strategy(self, rebalance_monthly=False):
        """Run Task 5 backtest over last year with optional monthly rebalancing.
        
        Args:
            rebalance_monthly: If True, rebalance monthly; if False, hold initial weights
            
        Returns:
            Dict with strategy and benchmark returns and cumulative performance
        """
        # Use last 252 trading days (approximately 1 year)
        backtest_data = self.returns.iloc[-252:].copy()
        
        if rebalance_monthly:
            strategy_returns = self._monthly_rebalancing(backtest_data)
        else:
            strategy_returns = backtest_data @ self.weights
            
        benchmark_returns = backtest_data @ self.benchmark_weights
        
        return {
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "strategy_cumulative": (1 + strategy_returns).cumprod(),
            "benchmark_cumulative": (1 + benchmark_returns).cumprod(),
        }
    
    def _monthly_rebalancing(self, data):
        """Simulate monthly rebalancing strategy."""
        monthly_returns = []
        for month_start in range(0, len(data), 21):  # ~21 trading days per month
            month_data = data.iloc[month_start:month_start+21]
            if len(month_data) > 0:
                month_returns = month_data @ self.weights
                monthly_returns.extend(month_returns)
        return pd.Series(monthly_returns, index=data.index[:len(monthly_returns)])

    def calculate_metrics(self, returns):
        """Calculate comprehensive performance metrics.

        Args:
            returns: Series of portfolio returns

        Returns:
            Dict with total return, annual return, volatility, Sharpe ratio, max drawdown
        """
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        cumulative = returns.cumsum()

        return {
            "total_return": (1 + returns).prod() - 1,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": annual_return / annual_vol,
            "max_drawdown": (cumulative - cumulative.expanding().max()).min(),
        }

    def run_task5_backtest(self, rebalance_monthly=False):
        """Execute complete Task 5 backtesting analysis.
        
        Args:
            rebalance_monthly: Whether to rebalance monthly or hold weights
            
        Returns:
            Performance comparison and analysis summary
        """
        results = self.backtest_strategy(rebalance_monthly)
        
        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(results["strategy_cumulative"], label="Optimized Strategy", linewidth=2.5, color="blue")
        ax.plot(results["benchmark_cumulative"], label="Benchmark (60% SPY / 40% BND)", linewidth=2.5, color="red")
        ax.set_title("Task 5: Strategy Backtesting - Last Year Performance", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Calculate and display metrics
        strategy_metrics = self.calculate_metrics(results["strategy_returns"])
        benchmark_metrics = self.calculate_metrics(results["benchmark_returns"])
        
        comparison = pd.DataFrame({
            "Strategy": strategy_metrics,
            "Benchmark": benchmark_metrics
        })
        
        # Performance analysis
        outperformed = strategy_metrics["total_return"] > benchmark_metrics["total_return"]
        better_sharpe = strategy_metrics["sharpe_ratio"] > benchmark_metrics["sharpe_ratio"]
        
        print("=" * 70)
        print("TASK 5: BACKTESTING RESULTS SUMMARY")
        print("=" * 70)
        print(f"Backtesting Period: Last 252 trading days (~1 year)")
        print(f"Strategy Outperformed Benchmark: {'YES' if outperformed else 'NO'}")
        print(f"Better Risk-Adjusted Returns: {'YES' if better_sharpe else 'NO'}")
        print("\nStrategy Viability Assessment:")
        
        if outperformed and better_sharpe:
            print("✓ STRONG: Strategy shows superior returns and risk-adjusted performance")
        elif outperformed:
            print("⚠ MODERATE: Strategy has higher returns but may carry more risk")
        elif better_sharpe:
            print("⚠ MODERATE: Strategy offers better risk-adjusted returns despite lower total returns")
        else:
            print("✗ WEAK: Strategy underperformed benchmark on both metrics")
            
        return comparison