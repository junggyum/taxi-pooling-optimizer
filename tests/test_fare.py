import unittest

from algorithms.route_optimizer import exact_route
from services.fare_service import FareSettings, calculate_pool_fares
from tests.helpers import two_passenger_problem


class FareTests(unittest.TestCase):
    def test_passenger_shares_sum_to_pool_charge(self):
        problem = two_passenger_problem()
        plan = exact_route(problem)
        individual_legs = {
            "A": problem.leg("A", "B"),
            "B": problem.leg("C", "D"),
        }

        result = calculate_pool_fares(problem.passengers, individual_legs, plan)

        self.assertEqual(sum(result.passenger_shares.values()), result.pooled_charge_total_won)
        self.assertEqual(result.discount_rate, 0.30)
        self.assertGreater(result.savings_total_won, 0)

    def test_discount_is_capped_and_negative_saving_becomes_zero(self):
        settings = FareSettings(max_discount=0.10)
        problem = two_passenger_problem()
        plan = exact_route(problem)
        individual_legs = {"A": problem.leg("A", "B"), "B": problem.leg("C", "D")}
        result = calculate_pool_fares(problem.passengers, individual_legs, plan, settings)
        self.assertEqual(result.discount_rate, 0.10)

        long_route = plan.__class__(
            start_passenger_id=plan.start_passenger_id,
            events=plan.events,
            total_distance_km=100.0,
            total_duration_min=plan.total_duration_min,
        )
        no_savings = calculate_pool_fares(problem.passengers, individual_legs, long_route)
        self.assertEqual(no_savings.distance_savings_rate, 0.0)
        self.assertEqual(no_savings.discount_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
