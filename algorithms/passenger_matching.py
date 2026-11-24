"""Candidate grouping and disjoint-group selection for shared rides."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import cos, radians, sqrt
import math
from typing import Mapping, Sequence

from algorithms.route_optimizer import RouteProblem, exact_route
from models.passenger import Passenger
from models.route import Leg, RoutePlan


@dataclass(frozen=True)
class MatchGroup:
    passengers: tuple[Passenger, ...]
    route_plan: RoutePlan
    individual_distance_km: float
    distance_savings_km: float
    distance_savings_rate: float
    pickup_compactness_km: float
    route_similarity: float

    @property
    def passenger_ids(self) -> tuple[str, ...]:
        return tuple(p.passenger_id for p in self.passengers)


@dataclass(frozen=True)
class MatchingResult:
    selected_groups: tuple[MatchGroup, ...]
    unmatched_passenger_ids: tuple[str, ...]
    feasible_candidate_count: int
    rejected_counts: dict[str, int]


def _direction(passenger: Passenger) -> tuple[float, float]:
    mean_lat = radians((passenger.pickup.latitude + passenger.dropoff.latitude) / 2)
    east = (passenger.dropoff.longitude - passenger.pickup.longitude) * cos(mean_lat)
    north = passenger.dropoff.latitude - passenger.pickup.latitude
    length = sqrt(east * east + north * north)
    if length == 0:
        return (0.0, 0.0)
    return east / length, north / length


def _pair_similarity(first: Passenger, second: Passenger) -> float:
    a_east, a_north = _direction(first)
    b_east, b_north = _direction(second)
    cosine = a_east * b_east + a_north * b_north
    return max(0.0, min(1.0, cosine))


def _pickup_distance(first: Passenger, second: Passenger, legs: Mapping[tuple[str, str], Leg]) -> float:
    a = first.pickup.location_id
    b = second.pickup.location_id
    distances = []
    for key in ((a, b), (b, a)):
        if key in legs:
            distances.append(legs[key].distance_km)
    return sum(distances) / len(distances) if distances else math.inf


def match_passengers(
    passengers: Sequence[Passenger],
    legs: Mapping[tuple[str, str], Leg],
    individual_legs: Mapping[str, Leg],
    max_capacity: int = 4,
    strategy: str = "efficiency",
    neighbor_limit: int = 6,
) -> MatchingResult:
    """Build bounded candidate groups, check each exact route, then select disjoint groups.

    For 20 requests, candidate generation considers each rider's nearest pickup
    neighbors to keep the search bounded. Group selection is greedy; it does not
    claim to solve the global set-packing problem optimally.
    """
    if strategy not in {"nearest", "similarity", "efficiency"}:
        raise ValueError("strategy must be nearest, similarity, or efficiency")
    if not 1 <= max_capacity <= 4:
        raise ValueError("max_capacity must be between 1 and 4")

    shareable = [p for p in passengers if p.share_allowed]
    if max_capacity < 2 or len(shareable) < 2:
        return MatchingResult((), tuple(p.passenger_id for p in passengers), 0, {})

    group_keys: set[tuple[str, ...]] = set()
    for passenger in shareable:
        others = [p for p in shareable if p.passenger_id != passenger.passenger_id]
        others.sort(key=lambda other: (_pickup_distance(passenger, other, legs), other.passenger_id))
        neighbors = others[: min(neighbor_limit, len(others))]
        for size in range(2, min(max_capacity, len(neighbors) + 1) + 1):
            for subset in combinations(neighbors, size - 1):
                key = tuple(sorted([passenger.passenger_id, *(p.passenger_id for p in subset)]))
                group_keys.add(key)

    by_id = {p.passenger_id: p for p in shareable}
    candidates: list[MatchGroup] = []
    rejected: dict[str, int] = {"no_feasible_route": 0, "no_distance_savings": 0}

    for key in sorted(group_keys):
        group = tuple(by_id[pid] for pid in key)
        problem = RouteProblem(
            passengers=group,
            legs=legs,
            solo_times_min={pid: individual_legs[pid].duration_min for pid in key},
            max_capacity=max_capacity,
        )
        plan = exact_route(problem)
        if plan is None:
            rejected["no_feasible_route"] += 1
            continue

        solo_distance = sum(individual_legs[pid].distance_km for pid in key)
        savings = solo_distance - plan.total_distance_km
        if savings <= 1e-6:
            rejected["no_distance_savings"] += 1
            continue
        pair_distances = [
            _pickup_distance(a, b, legs)
            for a, b in combinations(group, 2)
        ]
        pair_similarities = [_pair_similarity(a, b) for a, b in combinations(group, 2)]
        candidates.append(
            MatchGroup(
                passengers=group,
                route_plan=plan,
                individual_distance_km=solo_distance,
                distance_savings_km=savings,
                distance_savings_rate=savings / solo_distance if solo_distance else 0.0,
                pickup_compactness_km=sum(pair_distances) / len(pair_distances),
                route_similarity=sum(pair_similarities) / len(pair_similarities),
            )
        )

    if strategy == "nearest":
        candidates.sort(
            key=lambda g: (-g.pickup_compactness_km, g.distance_savings_km / len(g.passengers), g.passenger_ids),
            reverse=True,
        )
    elif strategy == "similarity":
        candidates.sort(
            key=lambda g: (g.route_similarity, g.distance_savings_km / len(g.passengers), g.passenger_ids),
            reverse=True,
        )
    else:
        candidates.sort(
            key=lambda g: (
                g.distance_savings_km / len(g.passengers),
                g.distance_savings_rate,
                len(g.passengers),
                g.passenger_ids,
            ),
            reverse=True,
        )

    selected: list[MatchGroup] = []
    used: set[str] = set()
    for candidate in candidates:
        if used.isdisjoint(candidate.passenger_ids):
            selected.append(candidate)
            used.update(candidate.passenger_ids)

    unmatched = tuple(p.passenger_id for p in passengers if p.passenger_id not in used)
    return MatchingResult(
        selected_groups=tuple(selected),
        unmatched_passenger_ids=unmatched,
        feasible_candidate_count=len(candidates),
        rejected_counts={key: value for key, value in rejected.items() if value},
    )
