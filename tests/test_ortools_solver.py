import unittest

from algorithms.ortools_solver import ortools_route
from tests.helpers import two_passenger_problem


class OrToolsTests(unittest.TestCase):
    def test_solver_returns_valid_pickup_delivery_plan(self):
        problem = two_passenger_problem()
        plan = ortools_route(problem, time_limit_seconds=1)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.feasible)
        self.assertEqual(set(plan.detour_minutes), {"A", "B"})
        self.assertFalse(plan.detour_violations)
        self.assertEqual(len(plan.events), 4)


if __name__ == "__main__":
    unittest.main()
