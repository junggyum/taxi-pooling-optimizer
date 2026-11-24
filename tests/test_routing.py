import unittest

from algorithms.greedy import greedy_route
from algorithms.route_optimizer import exact_route
from tests.helpers import two_passenger_problem


class RoutingTests(unittest.TestCase):
    def test_exact_route_respects_pickup_precedence_and_is_feasible(self):
        problem = two_passenger_problem()
        plan = exact_route(problem)

        self.assertIsNotNone(plan)
        self.assertTrue(plan.feasible)
        self.assertLessEqual(plan.total_distance_km, 7.0)
        positions = {
            (event.kind, event.passenger_id): index
            for index, event in enumerate(plan.events)
        }
        for passenger in problem.passengers:
            self.assertLess(
                positions[("pickup", passenger.passenger_id)],
                positions[("dropoff", passenger.passenger_id)],
            )

    def test_greedy_returns_a_constraint_valid_route(self):
        plan = greedy_route(two_passenger_problem())
        self.assertIsNotNone(plan)
        self.assertTrue(plan.feasible)
        self.assertEqual(set(plan.detour_minutes), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
