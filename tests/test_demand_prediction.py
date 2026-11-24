import unittest

from services.demand_prediction import predict_demo_demand, synthetic_training_data


class DemandPredictionTests(unittest.TestCase):
    def test_training_data_is_deterministic_and_explicitly_synthetic(self):
        first = synthetic_training_data(samples=200, seed=7)
        second = synthetic_training_data(samples=200, seed=7)
        self.assertTrue(first.equals(second))
        self.assertEqual(set(first.columns), {"hour", "weekday", "zone_code", "requests"})

    def test_forecast_is_reproducible_nonnegative_demo_value(self):
        first = predict_demo_demand(hour=18, weekday=2, zone_code=0, seed=7)
        second = predict_demo_demand(hour=18, weekday=2, zone_code=0, seed=7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.expected_requests, 0)
        self.assertIn("가상", first.label)


if __name__ == "__main__":
    unittest.main()
