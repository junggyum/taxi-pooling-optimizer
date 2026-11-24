"""A small, reproducible genetic search over valid pickup/dropoff orders."""

from __future__ import annotations

import random
import math
from typing import Sequence

from algorithms.route_optimizer import RouteProblem
from models.route import RoutePlan, VisitEvent


def _all_events(problem: RouteProblem, start_id: str) -> list[VisitEvent]:
    return [
        VisitEvent(kind, passenger.passenger_id)
        for passenger in problem.passengers
        for kind in ("pickup", "dropoff")
        if not (kind == "pickup" and passenger.passenger_id == start_id)
    ]


def _eligible_events(
    pending: set[VisitEvent],
    picked: set[str],
    active: set[str],
    capacity: int,
) -> list[VisitEvent]:
    eligible = []
    for event in pending:
        if event.kind == "pickup":
            if event.passenger_id not in picked and len(active) < capacity:
                eligible.append(event)
        elif event.passenger_id in active:
            eligible.append(event)
    return sorted(eligible, key=lambda event: (event.kind, event.passenger_id))


def _random_valid_sequence(problem: RouteProblem, start_id: str, rng: random.Random) -> list[VisitEvent]:
    pending = set(_all_events(problem, start_id))
    picked = {start_id}
    active = {start_id}
    sequence: list[VisitEvent] = []
    while pending:
        eligible = _eligible_events(pending, picked, active, problem.max_capacity)
        if not eligible:
            raise ValueError("could not generate a valid pickup/dropoff order")
        event = rng.choice(eligible)
        sequence.append(event)
        pending.remove(event)
        if event.kind == "pickup":
            picked.add(event.passenger_id)
            active.add(event.passenger_id)
        else:
            active.remove(event.passenger_id)
    return sequence


def _repair_order(problem: RouteProblem, start_id: str, order: Sequence[VisitEvent]) -> list[VisitEvent]:
    """Project a crossover/mutation result onto a valid precedence/capacity order."""
    rank = {event: index for index, event in enumerate(order)}
    pending = set(_all_events(problem, start_id))
    picked = {start_id}
    active = {start_id}
    repaired: list[VisitEvent] = []
    while pending:
        eligible = _eligible_events(pending, picked, active, problem.max_capacity)
        if not eligible:
            raise ValueError("order repair could not satisfy capacity constraints")
        event = min(eligible, key=lambda item: (rank.get(item, len(rank)), item.kind, item.passenger_id))
        repaired.append(event)
        pending.remove(event)
        if event.kind == "pickup":
            picked.add(event.passenger_id)
            active.add(event.passenger_id)
        else:
            active.remove(event.passenger_id)
    return repaired


def _order_crossover(
    first: Sequence[VisitEvent],
    second: Sequence[VisitEvent],
    rng: random.Random,
) -> list[VisitEvent]:
    if len(first) < 2:
        return list(first)
    left, right = sorted(rng.sample(range(len(first)), 2))
    child: list[VisitEvent | None] = [None] * len(first)
    child[left : right + 1] = first[left : right + 1]
    remaining = [event for event in second if event not in child]
    index = (right + 1) % len(child)
    for event in remaining:
        while child[index] is not None:
            index = (index + 1) % len(child)
        child[index] = event
    return [event for event in child if event is not None]


def _fitness(
    problem: RouteProblem,
    sequence: Sequence[VisitEvent],
    start_id: str,
) -> tuple[tuple[int, float], RoutePlan | None]:
    try:
        plan = problem.evaluate_sequence(sequence, start_id)
    except (KeyError, ValueError):
        return (1, math.inf), None
    excess = sum(plan.detour_violations.values())
    # Feasible candidates always outrank infeasible ones; distance is the main objective.
    score = plan.total_distance_km + excess * 25.0 + plan.total_duration_min * 0.001
    return ((0 if plan.feasible else 1, score), plan)


def genetic_route(
    problem: RouteProblem,
    population_size: int = 36,
    generations: int = 45,
    mutation_rate: float = 0.18,
    seed: int = 42,
) -> RoutePlan | None:
    """Search valid event orders with selection, order crossover, and swap mutation."""
    if population_size < 4 or generations < 1:
        raise ValueError("population_size must be at least 4 and generations at least 1")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")

    best: RoutePlan | None = None
    for start_index, start_id in enumerate(problem.passenger_by_id):
        rng = random.Random(seed + start_index)
        population = [
            _random_valid_sequence(problem, start_id, rng)
            for _ in range(population_size)
        ]
        elite_count = max(2, population_size // 8)

        for _ in range(generations):
            evaluated = [(*_fitness(problem, sequence, start_id), sequence) for sequence in population]
            evaluated.sort(key=lambda item: (item[0], tuple((e.kind, e.passenger_id) for e in item[2])))
            for _, plan, _ in evaluated[: max(4, population_size // 2)]:
                if plan is not None and plan.feasible and (
                    best is None
                    or (plan.total_distance_km, plan.total_duration_min)
                    < (best.total_distance_km, best.total_duration_min)
                ):
                    best = plan

            next_population = [list(sequence) for _, _, sequence in evaluated[:elite_count]]
            breeding_pool = evaluated[: max(4, population_size // 2)]
            while len(next_population) < population_size:
                first = rng.choice(breeding_pool)[2]
                second = rng.choice(breeding_pool)[2]
                child = _order_crossover(first, second, rng)
                if len(child) >= 2 and rng.random() < mutation_rate:
                    i, j = rng.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                next_population.append(_repair_order(problem, start_id, child))
            population = next_population

    return best
