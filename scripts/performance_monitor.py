#!/usr/bin/env python3
"""
Performance monitoring script for portfolio optimization
"""
import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

from data_loader import DataLoader
from portfolio_optimizer import PortfolioOptimizer
from config import Config

def monitor_performance():
    """Monitor performance during portfolio optimization"""
    
    print("=== Performance Monitoring Started ===")
    
    start_time = time.time()
    
    try:
        # Run portfolio optimization
        print("\nRunning portfolio optimization...")
        
        loader = DataLoader(Config.START_DATE, Config.END_DATE)
        raw_data = loader.fetch_data()
        processed_data = loader.preprocess_data(raw_data)
        returns_data = loader.get_combined_returns(processed_data)
        
        optimizer = PortfolioOptimizer(returns_data)
        
        # Time the optimization
        opt_start = time.time()
        optimal_weights = optimizer.optimize_portfolio('sharpe')
        opt_end = time.time()
        
        end_time = time.time()
        
        # Results
        print(f"\n=== Performance Results ===")
        print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
        print(f"Optimization Time: {opt_end - opt_start:.2f} seconds")
        
        # Save performance metrics
        performance_data = {
            'timestamp': datetime.now(),
            'total_time': end_time - start_time,
            'optimization_time': opt_end - opt_start
        }
        
        # Save to CSV
        results_path = Config.get_results_path('performance_metrics.csv')
        pd.DataFrame([performance_data]).to_csv(results_path, index=False)
        print(f"\nPerformance metrics saved to: {results_path}")
        
        return optimal_weights
        
    except Exception as e:
        print(f"Error during performance monitoring: {e}")
        return None

if __name__ == "__main__":
    monitor_performance()