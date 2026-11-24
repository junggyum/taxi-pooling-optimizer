"""Deterministic synthetic requests snapped to real Nowon road nodes."""

from __future__ import annotations

import random
from typing import Sequence

from config import MAX_DEMO_PASSENGERS, MIN_DEMO_PASSENGERS
from models.passenger import Passenger
from services.routing_service import MapDataError, RoadRouter, haversine_km


def generate_demo_passengers(
    router: RoadRouter,
    count: int = 8,
    seed: int = 42,
) -> list[Passenger]:
    if not MIN_DEMO_PASSENGERS <= count <= MAX_DEMO_PASSENGERS:
        raise ValueError(f"demo passenger count must be between {MIN_DEMO_PASSENGERS} and {MAX_DEMO_PASSENGERS}")

    rng = random.Random(seed)
    node_ids = list(router.graph.nodes)
    if len(node_ids) < count * 2:
        raise MapDataError("노원구 도로망 노드가 너무 적어 가상 승객을 만들 수 없습니다.")
    passengers: list[Passenger] = []
    max_attempts = count * 100

    for index in range(1, max_attempts + 1):
        if len(passengers) >= count:
            break
        origin_node, destination_node = rng.sample(node_ids, 2)
        origin_data = router.graph.nodes[origin_node]
        destination_data = router.graph.nodes[destination_node]
        straight_km = haversine_km(
            float(origin_data["y"]), float(origin_data["x"]),
            float(destination_data["y"]), float(destination_data["x"]),
        )
        if not 0.6 <= straight_km <= 6.0:
            continue

        passenger_number = len(passengers) + 1
        pickup = router.location_for_node(int(origin_node), f"가상 승객 {passenger_number:02d} 출발")
        dropoff = router.location_for_node(int(destination_node), f"가상 승객 {passenger_number:02d} 도착")
        try:
            direct_leg = router.route_between(pickup, dropoff)
        except MapDataError:
            continue
        if not 1.0 <= direct_leg.distance_km <= 10.0:
            continue

        passengers.append(
            Passenger(
                passenger_id=f"P{passenger_number:02d}",
                pickup=pickup,
                dropoff=dropoff,
                max_detour_min=rng.choice([0, 5, 10, 10, 15, 20]),
                requested_at=f"{rng.randrange(7, 23):02d}:{rng.choice(['00', '15', '30', '45'])}",
                share_allowed=True,
            )
        )

    if len(passengers) != count:
        raise MapDataError(
            f"가상 승객 {count}명 중 {len(passengers)}명만 생성했습니다. 노원구 도로망을 확인해 주세요."
        )
    return passengers
