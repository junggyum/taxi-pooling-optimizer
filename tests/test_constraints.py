import unittest

from algorithms.route_optimizer import RouteProblem, exact_route
from models.route import VisitEvent
from tests.helpers import make_legs, make_passenger, two_passenger_problem


class ConstraintTests(unittest.TestCase):
    def test_dropoff_before_pickup_is_rejected(self):
        problem = two_passenger_problem()
        with self.assertRaisesRegex(ValueError, "picked up before dropoff"):
            problem.evaluate_sequence(
                [
                    VisitEvent("dropoff", "B"),
                    VisitEvent("pickup", "B"),
                    VisitEvent("dropoff", "A"),
                ],
                start_passenger_id="A",
            )

    def test_passenger_group_cannot_exceed_capacity(self):
        with self.assertRaisesRegex(ValueError, "exceeds vehicle capacity"):
            two_passenger_problem(max_capacity=1)

    def test_zero_detour_limit_excludes_slower_shared_route(self):
        passengers = [
            make_passenger("A", "A", "B", detour=0),
            make_passenger("B", "C", "D", detour=0),
        ]
        problem = RouteProblem(
            passengers=passengers,
            legs=make_legs(),
            solo_times_min={"A": 5.0, "B": 5.0},
            max_capacity=2,
        )
        self.assertIsNone(exact_route(problem))


if __name__ == "__main__":
    unittest.main()
