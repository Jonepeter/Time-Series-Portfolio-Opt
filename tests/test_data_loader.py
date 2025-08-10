import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
import pandas as pd

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(start_date="2020-01-01", end_date="2020-12-31")
    
    def test_fetch_data(self):
        data = self.loader.fetch_data()
        self.assertEqual(len(data), 3)
        self.assertIn('TSLA', data)
        self.assertIn('BND', data)
        self.assertIn('SPY', data)
    
    def test_preprocess_data(self):
        raw_data = self.loader.fetch_data()
        processed_data = self.loader.preprocess_data(raw_data)
        
        for ticker, df in processed_data.items():
            self.assertIn('Daily_Return', df.columns)
            self.assertIn('Volatility', df.columns)
    
    def test_get_combined_returns(self):
        raw_data = self.loader.fetch_data()
        processed_data = self.loader.preprocess_data(raw_data)
        returns_df = self.loader.get_combined_returns(processed_data)
        
        self.assertEqual(len(returns_df.columns), 3)
        self.assertIsInstance(returns_df, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()