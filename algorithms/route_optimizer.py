"""Exact constrained search for small taxi pools (up to four passengers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from models.passenger import Passenger
from models.route import Leg, RoutePlan, VisitEvent


@dataclass(frozen=True)
class RouteProblem:
    passengers: Sequence[Passenger]
    legs: Mapping[tuple[str, str], Leg]
    solo_times_min: Mapping[str, float]
    max_capacity: int = 4

    def __post_init__(self) -> None:
        if not self.passengers:
            raise ValueError("a route problem needs at least one passenger")
        ids = [p.passenger_id for p in self.passengers]
        if len(set(ids)) != len(ids):
            raise ValueError("passenger IDs must be unique within a route problem")
        if not 1 <= self.max_capacity <= 4:
            raise ValueError("max_capacity must be between 1 and 4 in this MVP")
        if len(self.passengers) > self.max_capacity:
            raise ValueError("passenger group exceeds vehicle capacity")
        if set(ids) - set(self.solo_times_min):
            raise ValueError("solo travel times are required for every passenger")

    @property
    def passenger_by_id(self) -> dict[str, Passenger]:
        return {p.passenger_id: p for p in self.passengers}

    def leg(self, origin_id: str, destination_id: str) -> Leg:
        if origin_id == destination_id:
            return Leg(0.0, 0.0)
        try:
            return self.legs[(origin_id, destination_id)]
        except KeyError as exc:
            raise KeyError(f"missing road leg {origin_id!r} -> {destination_id!r}") from exc

    def stop_for(self, event: VisitEvent) -> str:
        passenger = self.passenger_by_id[event.passenger_id]
        return passenger.pickup.location_id if event.kind == "pickup" else passenger.dropoff.location_id

    def evaluate_sequence(
        self,
        events: Sequence[VisitEvent],
        start_passenger_id: str,
    ) -> RoutePlan:
        """Evaluate a stop sequence. The start rider boards at time zero."""
        passengers = self.passenger_by_id
        if start_passenger_id not in passengers:
            raise ValueError(f"unknown start passenger: {start_passenger_id}")

        anchor = passengers[start_passenger_id]
        expected = {
            VisitEvent(kind, passenger.passenger_id)
            for passenger in self.passengers
            for kind in ("pickup", "dropoff")
        }
        initial = VisitEvent("pickup", start_passenger_id)
        expected.remove(initial)
        event_list = list(events)
        if len(event_list) != len(expected) or set(event_list) != expected:
            raise ValueError("event sequence must contain every remaining pickup/dropoff exactly once")

        active = {start_passenger_id}
        picked_at = {start_passenger_id: 0.0}
        dropped: set[str] = set()
        rides: dict[str, float] = {}
        detours: dict[str, float] = {}
        violations: dict[str, float] = {}
        route_legs: list[Leg] = []
        full_events = [initial]
        current = anchor.pickup.location_id
        elapsed = 0.0
        distance = 0.0

        for event in event_list:
            if event.passenger_id not in passengers:
                raise ValueError(f"unknown passenger in event: {event.passenger_id}")
            if event.kind == "pickup":
                if event.passenger_id in picked_at or event.passenger_id in dropped:
                    raise ValueError("a passenger cannot be picked up twice")
                if len(active) >= self.max_capacity:
                    raise ValueError("vehicle capacity would be exceeded")
            elif event.passenger_id not in active:
                raise ValueError("a passenger must be picked up before dropoff")

            next_stop = self.stop_for(event)
            leg = self.leg(current, next_stop)
            elapsed += leg.duration_min
            distance += leg.distance_km
            route_legs.append(leg)
            full_events.append(event)
            current = next_stop

            if event.kind == "pickup":
                active.add(event.passenger_id)
                picked_at[event.passenger_id] = elapsed
            else:
                active.remove(event.passenger_id)
                dropped.add(event.passenger_id)
                ride = elapsed - picked_at[event.passenger_id]
                detour = max(0.0, ride - self.solo_times_min[event.passenger_id])
                rides[event.passenger_id] = ride
                detours[event.passenger_id] = detour
                excess = detour - passengers[event.passenger_id].max_detour_min
                if excess > 1e-9:
                    violations[event.passenger_id] = excess

        if active or len(dropped) != len(passengers):
            raise ValueError("every passenger must be dropped off before the route ends")

        return RoutePlan(
            start_passenger_id=start_passenger_id,
            events=tuple(full_events),
            total_distance_km=distance,
            total_duration_min=elapsed,
            passenger_ride_minutes=rides,
            detour_minutes=detours,
            detour_violations=violations,
            legs=tuple(route_legs),
            feasible=not violations,
        )


def exact_route(problem: RouteProblem) -> RoutePlan | None:
    """Enumerate precedence- and capacity-valid orders and return the shortest feasible route."""
    best: RoutePlan | None = None
    passengers = problem.passenger_by_id

    for start_id, anchor in passengers.items():
        remaining = [
            VisitEvent(kind, passenger.passenger_id)
            for passenger in problem.passengers
            for kind in ("pickup", "dropoff")
            if not (kind == "pickup" and passenger.passenger_id == start_id)
        ]

        def visit(
            pending: tuple[VisitEvent, ...],
            current: str,
            elapsed: float,
            distance: float,
            active: frozenset[str],
            picked_at: dict[str, float],
            sequence: tuple[VisitEvent, ...],
        ) -> None:
            nonlocal best
            if best is not None and distance > best.total_distance_km + 1e-9:
                return
            if not pending:
                try:
                    plan = problem.evaluate_sequence(sequence[1:], start_id)
                except ValueError:
                    return
                if plan.feasible and (
                    best is None
                    or (plan.total_distance_km, plan.total_duration_min)
                    < (best.total_distance_km, best.total_duration_min)
                ):
                    best = plan
                return

            choices: list[tuple[float, str, str, VisitEvent, Leg]] = []
            for event in pending:
                if event.kind == "pickup":
                    if event.passenger_id in picked_at or len(active) >= problem.max_capacity:
                        continue
                elif event.passenger_id not in active:
                    continue
                try:
                    leg = problem.leg(current, problem.stop_for(event))
                except KeyError:
                    # A one-way or disconnected road pair is simply not a legal next stop.
                    continue
                choices.append((leg.distance_km, event.kind, event.passenger_id, event, leg))

            for _, _, _, event, leg in sorted(choices):
                next_elapsed = elapsed + leg.duration_min
                if event.kind == "dropoff":
                    ride = next_elapsed - picked_at[event.passenger_id]
                    detour = max(0.0, ride - problem.solo_times_min[event.passenger_id])
                    if detour > passengers[event.passenger_id].max_detour_min + 1e-9:
                        continue
                    next_active = active - {event.passenger_id}
                else:
                    next_active = active | {event.passenger_id}

                next_picked = picked_at
                if event.kind == "pickup":
                    next_picked = {**picked_at, event.passenger_id: next_elapsed}
                next_pending = tuple(item for item in pending if item != event)
                visit(
                    next_pending,
                    problem.stop_for(event),
                    next_elapsed,
                    distance + leg.distance_km,
                    frozenset(next_active),
                    next_picked,
                    sequence + (event,),
                )

        initial = VisitEvent("pickup", start_id)
        visit(
            tuple(remaining),
            anchor.pickup.location_id,
            0.0,
            0.0,
            frozenset({start_id}),
            {start_id: 0.0},
            (initial,),
        )

    return best
