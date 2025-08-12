import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from forecasting_models import ForecastingModels


class TestForecastingModels(unittest.TestCase):
    def setUp(self):
        # Create sample processed data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        self.processed_data = {
            'TSLA': pd.DataFrame({
                'Close': np.random.normal(100, 10, 252),
                'Daily_Return': np.random.normal(0.001, 0.03, 252),
                'Volatility': np.random.normal(0.02, 0.005, 252)
            }, index=dates)
        }
        
        self.forecaster = ForecastingModels(self.processed_data, 'TSLA')

    def test_initialization(self):
        self.assertEqual(self.forecaster.ticker, 'TSLA')
        self.assertIn('TSLA', self.forecaster.data)

    def test_prepare_data(self):
        train_data, test_data = self.forecaster.prepare_data()
        
        self.assertIsInstance(train_data, pd.Series)
        self.assertIsInstance(test_data, pd.Series)
        self.assertGreater(len(train_data), len(test_data))

    def test_fit_arima_model(self):
        train_data, _ = self.forecaster.prepare_data()
        model = self.forecaster.fit_arima_model(train_data)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forecast'))

    def test_evaluate_models(self):
        results = self.forecaster.evaluate_models()
        
        required_keys = ['arima_mae', 'arima_rmse', 'lstm_mae', 'lstm_rmse', 'arima_model']
        for key in required_keys:
            self.assertIn(key, results)
        
        self.assertGreater(results['arima_mae'], 0)
        self.assertGreater(results['arima_rmse'], 0)


if __name__ == "__main__":
    unittest.main()