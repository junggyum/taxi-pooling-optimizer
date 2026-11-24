"""Coordinates matching, route solvers, virtual fares, and result metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

from algorithms.genetic import genetic_route
from algorithms.greedy import greedy_route
from algorithms.ortools_solver import ortools_route
from algorithms.passenger_matching import MatchGroup, MatchingResult, match_passengers
from algorithms.route_optimizer import RouteProblem, exact_route
from models.passenger import Passenger
from models.route import Leg, RoutePlan
from services.fare_service import FareResult, FareSettings, calculate_pool_fares, estimate_fare
from services.routing_service import RoadRouter, individual_route_legs


@dataclass(frozen=True)
class AlgorithmOutcome:
    name: str
    plan: RoutePlan | None
    runtime_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class GroupOutcome:
    match_group: MatchGroup
    individual_legs: dict[str, Leg]
    selected_plan: RoutePlan
    fare_result: FareResult
    algorithm_results: dict[str, AlgorithmOutcome]


@dataclass(frozen=True)
class OptimizationResult:
    passengers: tuple[Passenger, ...]
    matching: MatchingResult
    individual_legs: dict[str, Leg]
    group_outcomes: tuple[GroupOutcome, ...]
    max_capacity: int
    matching_strategy: str
    fare_settings: FareSettings

    @property
    def pooled_ids(self) -> set[str]:
        return {
            passenger.passenger_id
            for group in self.group_outcomes
            for passenger in group.match_group.passengers
        }

    def passenger_summary_rows(self) -> list[dict[str, object]]:
        group_for_id = {
            passenger.passenger_id: outcome
            for outcome in self.group_outcomes
            for passenger in outcome.match_group.passengers
        }
        rows: list[dict[str, object]] = []
        for passenger in self.passengers:
            solo = self.individual_legs[passenger.passenger_id]
            outcome = group_for_id.get(passenger.passenger_id)
            if outcome is None:
                individual_fare = estimate_fare(solo.distance_km, solo.duration_min, self.fare_settings)
                rows.append({
                    "승객": passenger.passenger_id,
                    "출발지": passenger.pickup.label,
                    "목적지": passenger.dropoff.label,
                    "결과": "개별 탑승",
                    "개별 거리(km)": round(solo.distance_km, 2),
                    "예상 시간(분)": round(solo.duration_min, 1),
                    "우회(분)": 0.0,
                    "허용 우회(분)": passenger.max_detour_min,
                    "개별 예상 요금(원)": individual_fare,
                    "합승 부담 요금(원)": individual_fare,
                    "절약(원)": 0,
                })
                continue

            fare = outcome.fare_result
            plan = outcome.selected_plan
            ride = plan.passenger_ride_minutes[passenger.passenger_id]
            detour = plan.detour_minutes[passenger.passenger_id]
            individual_fare = fare.individual_fares[passenger.passenger_id]
            shared_fare = fare.passenger_shares[passenger.passenger_id]
            rows.append({
                "승객": passenger.passenger_id,
                "출발지": passenger.pickup.label,
                "목적지": passenger.dropoff.label,
                "결과": "합승",
                "개별 거리(km)": round(solo.distance_km, 2),
                "예상 시간(분)": round(ride, 1),
                "우회(분)": round(detour, 1),
                "허용 우회(분)": passenger.max_detour_min,
                "개별 예상 요금(원)": individual_fare,
                "합승 부담 요금(원)": shared_fare,
                "절약(원)": individual_fare - shared_fare,
            })
        return rows

    def summary_metrics(self) -> dict[str, float | int | None]:
        individual_distance = sum(leg.distance_km for leg in self.individual_legs.values())
        individual_time = sum(leg.duration_min for leg in self.individual_legs.values())
        individual_fare = sum(
            estimate_fare(leg.distance_km, leg.duration_min, self.fare_settings)
            for leg in self.individual_legs.values()
        )
        pooled_distance = sum(
            outcome.selected_plan.total_distance_km for outcome in self.group_outcomes
        )
        # Passenger time is compared like-for-like; vehicle time is reported separately.
        pooled_time = sum(
            sum(outcome.selected_plan.passenger_ride_minutes.values())
            for outcome in self.group_outcomes
        )
        pooled_vehicle_time = sum(
            outcome.selected_plan.total_duration_min for outcome in self.group_outcomes
        )
        pooled_charge = sum(
            outcome.fare_result.pooled_charge_total_won for outcome in self.group_outcomes
        )
        solo_ids = set(self.individual_legs) - self.pooled_ids
        for passenger_id in solo_ids:
            solo = self.individual_legs[passenger_id]
            pooled_distance += solo.distance_km
            pooled_time += solo.duration_min
            solo_fare = estimate_fare(solo.distance_km, solo.duration_min, self.fare_settings)
            pooled_charge += solo_fare
        saved_distance = max(0.0, individual_distance - pooled_distance)
        total_passengers = len(self.passengers)
        pool_count = len(self.pooled_ids)
        shareable_count = sum(p.share_allowed for p in self.passengers)
        detour_records = [
            (passenger, outcome.selected_plan.detour_minutes[passenger.passenger_id])
            for outcome in self.group_outcomes
            for passenger in outcome.match_group.passengers
        ]
        detour_compliance_rate = (
            sum(detour <= passenger.max_detour_min + 1e-9 for passenger, detour in detour_records)
            / len(detour_records)
            if detour_records
            else None
        )
        return {
            "individual_distance_km": individual_distance,
            "pooled_distance_km": pooled_distance,
            "distance_saved_km": saved_distance,
            "distance_savings_rate": saved_distance / individual_distance if individual_distance else 0.0,
            "individual_time_min": individual_time,
            "pooled_passenger_time_min": pooled_time,
            "pooled_vehicle_operating_time_min": pooled_vehicle_time + sum(
                self.individual_legs[pid].duration_min for pid in solo_ids
            ),
            "individual_fare_won": individual_fare,
            "pooled_charge_won": pooled_charge,
            "fare_saved_won": individual_fare - pooled_charge,
            "fare_savings_rate": (individual_fare - pooled_charge) / individual_fare if individual_fare else 0.0,
            "matching_success_rate": pool_count / shareable_count if shareable_count else 0.0,
            "max_detour_compliance_rate": detour_compliance_rate,
            "pooled_passenger_count": pool_count,
            "passenger_count": total_passengers,
        }


def _time_algorithm(name: str, func, problem: RouteProblem) -> AlgorithmOutcome:
    started = perf_counter()
    try:
        plan = func(problem)
        return AlgorithmOutcome(name, plan, perf_counter() - started)
    except Exception as exc:
        return AlgorithmOutcome(name, None, perf_counter() - started, f"{type(exc).__name__}: {exc}")


def optimize_passengers(
    passengers: Sequence[Passenger],
    router: RoadRouter,
    max_capacity: int = 4,
    matching_strategy: str = "efficiency",
    fare_settings: FareSettings | None = None,
) -> OptimizationResult:
    if not passengers:
        raise ValueError("최적화할 승객이 없습니다.")
    fare_settings = fare_settings or FareSettings()
    leg_matrix = router.build_leg_matrix(passengers)
    individual_legs = individual_route_legs(passengers, leg_matrix)
    matching = match_passengers(
        passengers,
        leg_matrix,
        individual_legs,
        max_capacity=max_capacity,
        strategy=matching_strategy,
    )

    outcomes: list[GroupOutcome] = []
    for match_group in matching.selected_groups:
        group_passengers = match_group.passengers
        stop_ids = {
            location_id
            for passenger in group_passengers
            for location_id in (passenger.pickup.location_id, passenger.dropoff.location_id)
        }
        group_legs = {
            key: leg for key, leg in leg_matrix.items()
            if key[0] in stop_ids and key[1] in stop_ids
        }
        problem = RouteProblem(
            passengers=group_passengers,
            legs=group_legs,
            solo_times_min={pid: leg.duration_min for pid, leg in individual_legs.items() if pid in match_group.passenger_ids},
            max_capacity=max_capacity,
        )

        exact_result = _time_algorithm("Exact search", exact_route, problem)
        greedy_result = _time_algorithm("Greedy", greedy_route, problem)
        genetic_result = _time_algorithm("Genetic algorithm", genetic_route, problem)
        ortools_result = _time_algorithm(
            "OR-Tools", lambda route_problem: ortools_route(route_problem, time_limit_seconds=1), problem
        )
        algorithm_results = {
            result.name: result
            for result in (greedy_result, genetic_result, ortools_result, exact_result)
        }
        selected_plan = exact_result.plan or match_group.route_plan
        group_solo_legs = {pid: individual_legs[pid] for pid in match_group.passenger_ids}
        fares = calculate_pool_fares(group_passengers, group_solo_legs, selected_plan, fare_settings)
        outcomes.append(
            GroupOutcome(
                match_group=match_group,
                individual_legs=group_solo_legs,
                selected_plan=selected_plan,
                fare_result=fares,
                algorithm_results=algorithm_results,
            )
        )

    return OptimizationResult(
        passengers=tuple(passengers),
        matching=matching,
        individual_legs=individual_legs,
        group_outcomes=tuple(outcomes),
        max_capacity=max_capacity,
        matching_strategy=matching_strategy,
        fare_settings=fare_settings,
    )
