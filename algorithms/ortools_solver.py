"""OR-Tools pickup-and-delivery solver used as an independent comparison."""

from __future__ import annotations

import math

from algorithms.route_optimizer import RouteProblem
from models.route import RoutePlan, VisitEvent


def ortools_route(problem: RouteProblem, time_limit_seconds: int = 2) -> RoutePlan | None:
    """Solve one pooled group with pickup precedence, capacity, and ride-time limits."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete installs
        raise RuntimeError("OR-Tools is not installed; install the requirements.txt dependencies") from exc

    best: RoutePlan | None = None
    passenger_by_id = problem.passenger_by_id

    for start_id, anchor in passenger_by_id.items():
        # The taxi starts at the anchor rider's pickup with that rider already
        # onboard. That rider therefore only needs a dropoff event in the model.
        node_specs: list[tuple[str, str, str]] = [("start", start_id, "start")]
        node_specs.append(("event", start_id, "dropoff"))
        for passenger in problem.passengers:
            if passenger.passenger_id == start_id:
                continue
            node_specs.append(("event", passenger.passenger_id, "pickup"))
            node_specs.append(("event", passenger.passenger_id, "dropoff"))
        end_node = len(node_specs)
        node_specs.append(("end", start_id, "dropoff"))

        def location_id(node: int) -> str:
            if node == end_node:
                raise ValueError("the dummy end node has no geographic location")
            _, pid, kind = node_specs[node]
            if kind in {"start", "pickup"}:
                return passenger_by_id[pid].pickup.location_id
            return passenger_by_id[pid].dropoff.location_id

        manager = pywrapcp.RoutingIndexManager(len(node_specs), 1, [0], [end_node])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            if manager.IndexToNode(to_index) == end_node:
                return 0
            origin = location_id(manager.IndexToNode(from_index))
            destination = location_id(manager.IndexToNode(to_index))
            leg = problem.legs.get((origin, destination))
            return int(round(leg.distance_km * 1000)) if leg else 10**12

        def time_callback(from_index: int, to_index: int) -> int:
            if manager.IndexToNode(to_index) == end_node:
                return 0
            origin = location_id(manager.IndexToNode(from_index))
            destination = location_id(manager.IndexToNode(to_index))
            leg = problem.legs.get((origin, destination))
            return int(round(leg.duration_min * 60)) if leg else 10**12

        def demand_callback(index: int) -> int:
            node = manager.IndexToNode(index)
            if node == 0:
                return 1  # the anchor passenger is onboard as the taxi departs
            if node == end_node:
                return 0
            kind = node_specs[node][2]
            if kind == "pickup":
                return 1
            if kind == "dropoff":
                return -1
            return 0

        distance_index = routing.RegisterTransitCallback(distance_callback)
        time_index = routing.RegisterTransitCallback(time_callback)
        demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_index)
        routing.AddDimension(time_index, 0, 24 * 60 * 60, True, "Time")
        time_dimension = routing.GetDimensionOrDie("Time")
        routing.AddDimensionWithVehicleCapacity(
            demand_index,
            0,
            [problem.max_capacity],
            True,
            "Capacity",
        )

        pickup_nodes: dict[str, int] = {}
        dropoff_nodes: dict[str, int] = {}
        for node, (_, pid, kind) in enumerate(node_specs):
            if node == 0 or node == end_node:
                continue
            if kind == "pickup":
                pickup_nodes[pid] = node
            else:
                dropoff_nodes[pid] = node

        solver = routing.solver()
        for passenger in problem.passengers:
            dropoff_index = manager.NodeToIndex(dropoff_nodes[passenger.passenger_id])
            max_ride_seconds = math.ceil(
                (problem.solo_times_min[passenger.passenger_id] + passenger.max_detour_min) * 60
            )
            if passenger.passenger_id == start_id:
                # The anchor rider's trip clock starts at the route's time zero.
                solver.Add(time_dimension.CumulVar(dropoff_index) <= max_ride_seconds)
            else:
                pickup_index = manager.NodeToIndex(pickup_nodes[passenger.passenger_id])
                routing.AddPickupAndDelivery(pickup_index, dropoff_index)
                solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index))
                solver.Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(dropoff_index))
                solver.Add(
                    time_dimension.CumulVar(dropoff_index)
                    <= time_dimension.CumulVar(pickup_index) + max_ride_seconds
                )

        search = pywrapcp.DefaultRoutingSearchParameters()
        search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search.time_limit.seconds = max(1, int(time_limit_seconds))
        assignment = routing.SolveWithParameters(search)
        if assignment is None:
            continue

        sequence: list[VisitEvent] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                _, pid, kind = node_specs[node]
                sequence.append(VisitEvent(kind, pid))
            index = assignment.Value(routing.NextVar(index))

        try:
            plan = problem.evaluate_sequence(sequence, start_id)
        except (KeyError, ValueError):
            # The solver sees disconnected pairs as expensive arcs. Reject any
            # resulting path that still tries to traverse an unreachable road leg.
            continue
        if plan.feasible and (
            best is None
            or (plan.total_distance_km, plan.total_duration_min)
            < (best.total_distance_km, best.total_duration_min)
        ):
            best = plan

    return best
