"""Nearest-next-stop route ordering with pickup, capacity, and detour checks."""

from __future__ import annotations

from models.route import RoutePlan, VisitEvent
from algorithms.route_optimizer import RouteProblem


def greedy_route(problem: RouteProblem) -> RoutePlan | None:
    best: RoutePlan | None = None
    passengers = problem.passenger_by_id

    for start_id, anchor in passengers.items():
        pending = {
            VisitEvent(kind, p.passenger_id)
            for p in problem.passengers
            for kind in ("pickup", "dropoff")
            if not (kind == "pickup" and p.passenger_id == start_id)
        }
        active = {start_id}
        picked_at = {start_id: 0.0}
        current = anchor.pickup.location_id
        elapsed = 0.0
        sequence: list[VisitEvent] = []
        failed = False

        while pending:
            choices: list[tuple[float, int, str, VisitEvent, float]] = []
            for event in pending:
                if event.kind == "pickup":
                    if event.passenger_id in picked_at or len(active) >= problem.max_capacity:
                        continue
                elif event.passenger_id not in active:
                    continue

                try:
                    leg = problem.leg(current, problem.stop_for(event))
                except KeyError:
                    continue
                next_elapsed = elapsed + leg.duration_min
                if event.kind == "dropoff":
                    ride = next_elapsed - picked_at[event.passenger_id]
                    detour = max(0.0, ride - problem.solo_times_min[event.passenger_id])
                    if detour > passengers[event.passenger_id].max_detour_min + 1e-9:
                        continue
                # Prefer a dropoff on equal distance to release a seat sooner.
                priority = 0 if event.kind == "dropoff" else 1
                choices.append((leg.distance_km, priority, event.passenger_id, event, leg.duration_min))

            if not choices:
                failed = True
                break
            _, _, _, event, duration = min(choices)
            elapsed += duration
            current = problem.stop_for(event)
            pending.remove(event)
            sequence.append(event)
            if event.kind == "pickup":
                active.add(event.passenger_id)
                picked_at[event.passenger_id] = elapsed
            else:
                active.remove(event.passenger_id)

        if failed:
            continue
        plan = problem.evaluate_sequence(sequence, start_id)
        if plan.feasible and (
            best is None
            or (plan.total_distance_km, plan.total_duration_min)
            < (best.total_distance_km, best.total_duration_min)
        ):
            best = plan
    return best
