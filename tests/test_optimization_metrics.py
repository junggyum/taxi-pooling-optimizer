import unittest

from algorithms.passenger_matching import MatchGroup, MatchingResult
from models.passenger import Location, Passenger
from models.route import Leg, RoutePlan, VisitEvent
from services.fare_service import calculate_pool_fares, FareSettings
from services.optimization_service import GroupOutcome, OptimizationResult


def passenger(passenger_id: str, detour: int = 10, share_allowed: bool = True) -> Passenger:
    return Passenger(
        passenger_id,
        Location(f"{passenger_id}_PU", f"{passenger_id} 출발", 37.65, 127.05),
        Location(f"{passenger_id}_DO", f"{passenger_id} 도착", 37.66, 127.06),
        max_detour_min=detour,
        share_allowed=share_allowed,
    )


class OptimizationMetricTests(unittest.TestCase):
    def test_unmatched_passenger_time_is_added_once(self):
        first, second, solo = passenger("A"), passenger("B"), passenger("C", share_allowed=False)
        individual_legs = {
            "A": Leg(10.0, 10.0),
            "B": Leg(10.0, 12.0),
            "C": Leg(5.0, 5.0),
        }
        plan = RoutePlan(
            start_passenger_id="A",
            events=(
                VisitEvent("pickup", "A"),
                VisitEvent("pickup", "B"),
                VisitEvent("dropoff", "B"),
                VisitEvent("dropoff", "A"),
            ),
            total_distance_km=15.0,
            total_duration_min=15.0,
            passenger_ride_minutes={"A": 10.5, "B": 12.0},
            detour_minutes={"A": 0.5, "B": 0.0},
        )
        group = MatchGroup((first, second), plan, 20.0, 5.0, 0.25, 1.0, 0.9)
        fare = calculate_pool_fares((first, second), {"A": individual_legs["A"], "B": individual_legs["B"]}, plan)
        outcome = GroupOutcome(group, {"A": individual_legs["A"], "B": individual_legs["B"]}, plan, fare, {})
        result = OptimizationResult(
            passengers=(first, second, solo),
            matching=MatchingResult((group,), ("C",), 1, {}),
            individual_legs=individual_legs,
            group_outcomes=(outcome,),
            max_capacity=4,
            matching_strategy="efficiency",
            fare_settings=FareSettings(),
        )

        metrics = result.summary_metrics()
        self.assertEqual(metrics["pooled_passenger_time_min"], 10.5 + 12.0 + 5.0)
        self.assertEqual(metrics["pooled_vehicle_operating_time_min"], 15.0 + 5.0)
        self.assertEqual(metrics["max_detour_compliance_rate"], 1.0)
        self.assertGreaterEqual(metrics["fare_savings_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
