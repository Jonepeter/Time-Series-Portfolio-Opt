import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory.

    Args:
        returns_data: DataFrame of daily returns for assets
        tsla_forecast: Optional forecasted annual return for TSLA
    """

    def __init__(self, returns_data, tsla_forecast=None):
        self.returns = returns_data
        self.cov_matrix = returns_data.cov() * 252

        # Use forecasted return for TSLA if provided, otherwise use historical
        if tsla_forecast is not None:
            self.mean_returns = returns_data.mean() * 252
            self.mean_returns["TSLA"] = (
                tsla_forecast  # Assume forecast is already annualized
            )
        else:
            self.mean_returns = returns_data.mean() * 252

    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility."""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_std

    def negative_sharpe(self, weights, risk_free_rate=0.02):
        """Calculate negative Sharpe ratio for optimization."""
        p_return, p_std = self.portfolio_performance(weights)
        return -(p_return - risk_free_rate) / p_std

    def minimize_volatility(self, weights):
        """Return portfolio volatility for minimization."""
        return self.portfolio_performance(weights)[1]

    def optimize_portfolio(self, target="sharpe"):
        """Optimize portfolio for maximum Sharpe ratio or minimum volatility."""
        num_assets = len(self.mean_returns)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1.0 / num_assets]

        if target == "sharpe":
            result = minimize(
                self.negative_sharpe,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
        else:
            result = minimize(
                self.minimize_volatility,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

        return result.x

    def efficient_frontier(self, num_portfolios=100):
        """Generate efficient frontier data points."""
        results = np.zeros((3, num_portfolios))
        target_returns = np.linspace(
            self.mean_returns.min(), self.mean_returns.max(), num_portfolios
        )

        for i, target in enumerate(target_returns):
            constraints = (
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: np.sum(self.mean_returns * x) - target},
            )
            bounds = tuple((0, 1) for _ in range(len(self.mean_returns)))
            initial_guess = len(self.mean_returns) * [1.0 / len(self.mean_returns)]

            result = minimize(
                self.minimize_volatility,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                results[0, i] = target
                results[1, i] = result.fun
                results[2, i] = (target - 0.02) / result.fun

        return results

    def plot_efficient_frontier(self):
        """Plot efficient frontier with optimal portfolios marked."""
        frontier = self.efficient_frontier()

        plt.figure(figsize=(12, 8))
        plt.scatter(frontier[1], frontier[0], c=frontier[2], cmap="viridis", alpha=0.7)
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel("Portfolio Volatility (Risk)")
        plt.ylabel("Portfolio Return")
        plt.title("Efficient Frontier - Portfolio Optimization")

        # Calculate optimal portfolios
        max_sharpe_weights = self.optimize_portfolio("sharpe")
        min_vol_weights = self.optimize_portfolio("volatility")

        max_sharpe_return, max_sharpe_vol = self.portfolio_performance(
            max_sharpe_weights
        )
        min_vol_return, min_vol_vol = self.portfolio_performance(min_vol_weights)

        # Mark optimal portfolios
        plt.scatter(
            max_sharpe_vol,
            max_sharpe_return,
            marker="*",
            color="red",
            s=500,
            label="Maximum Sharpe Ratio Portfolio",
            edgecolors="black",
            linewidth=2,
        )
        plt.scatter(
            min_vol_vol,
            min_vol_return,
            marker="*",
            color="green",
            s=500,
            label="Minimum Volatility Portfolio",
            edgecolors="black",
            linewidth=2,
        )

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return max_sharpe_weights, min_vol_weights

    def analyze_portfolio(self, weights, portfolio_name="Portfolio"):
        """Analyze and display portfolio metrics."""
        portfolio_return, portfolio_vol = self.portfolio_performance(weights)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol

        print(f"\n{portfolio_name} Analysis:")
        print(f"Expected Annual Return: {portfolio_return:.2%}")
        print(f"Annual Volatility: {portfolio_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Asset Allocation:")
        for i, asset in enumerate(self.returns.columns):
            print(f"  {asset}: {weights[i]:.1%}")

        return portfolio_return, portfolio_vol, sharpe_ratio

    def recommend_portfolio(self):
        """Generate portfolio recommendation based on analysis."""
        max_sharpe_weights = self.optimize_portfolio("sharpe")
        min_vol_weights = self.optimize_portfolio("volatility")

        print("=" * 60)
        print("PORTFOLIO OPTIMIZATION RESULTS")
        print("=" * 60)

        # Analyze both portfolios
        max_sharpe_metrics = self.analyze_portfolio(
            max_sharpe_weights, "Maximum Sharpe Ratio"
        )
        min_vol_metrics = self.analyze_portfolio(min_vol_weights, "Minimum Volatility")

        # Recommendation logic
        print("\n" + "=" * 60)
        print("PORTFOLIO RECOMMENDATION")
        print("=" * 60)

        if max_sharpe_metrics[2] > 1.0:  # Sharpe ratio > 1.0
            recommended_weights = max_sharpe_weights
            print("\nRECOMMENDED: Maximum Sharpe Ratio Portfolio")
            print("Justification: High risk-adjusted returns with Sharpe ratio > 1.0")
        else:
            recommended_weights = min_vol_weights
            print("\nRECOMMENDED: Minimum Volatility Portfolio")
            print("Justification: Lower risk approach given modest Sharpe ratios")

        print("\nFinal Portfolio Summary:")
        self.analyze_portfolio(recommended_weights, "Recommended")

        return recommended_weights

    def plot_asset_allocations(self):
        """Plot pie charts of asset allocations for min vol and max Sharpe portfolios."""
        max_sharpe_weights = self.optimize_portfolio("sharpe")
        min_vol_weights = self.optimize_portfolio("volatility")
        assets = self.returns.columns
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.pie(min_vol_weights, labels=assets, autopct="%1.1f%%", startangle=90)
        plt.title("Minimum Volatility Portfolio Allocation")
        plt.subplot(1, 2, 2)
        plt.pie(max_sharpe_weights, labels=assets, autopct="%1.1f%%", startangle=90)
        plt.title("Maximum Sharpe Ratio Portfolio Allocation")
        plt.tight_layout()
        plt.show()
