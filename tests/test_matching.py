import unittest

from algorithms.passenger_matching import match_passengers
from tests.helpers import two_passenger_problem


class MatchingTests(unittest.TestCase):
    def test_efficiency_strategy_selects_feasible_saving_group(self):
        problem = two_passenger_problem()
        individual_legs = {
            p.passenger_id: problem.leg(p.pickup.location_id, p.dropoff.location_id)
            for p in problem.passengers
        }
        result = match_passengers(
            problem.passengers,
            problem.legs,
            individual_legs,
            max_capacity=2,
            strategy="efficiency",
        )

        self.assertEqual(len(result.selected_groups), 1)
        self.assertEqual(set(result.selected_groups[0].passenger_ids), {"A", "B"})
        self.assertEqual(result.unmatched_passenger_ids, ())

    def test_invalid_strategy_is_rejected(self):
        problem = two_passenger_problem()
        individual_legs = {"A": problem.leg("A", "B"), "B": problem.leg("C", "D")}
        with self.assertRaisesRegex(ValueError, "strategy must be"):
            match_passengers(problem.passengers, problem.legs, individual_legs, strategy="random")


if __name__ == "__main__":
    unittest.main()
