import unittest

from algorithms.genetic import genetic_route
from tests.helpers import two_passenger_problem


class GeneticTests(unittest.TestCase):
    def test_genetic_search_returns_feasible_route(self):
        plan = genetic_route(
            two_passenger_problem(),
            population_size=16,
            generations=18,
            seed=3,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan.feasible)
        self.assertEqual(len(plan.events), 4)


if __name__ == "__main__":
    unittest.main()
