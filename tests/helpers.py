from models.passenger import Location, Passenger
from models.route import Leg


def make_passenger(passenger_id: str, pickup: str, dropoff: str, detour: int = 10) -> Passenger:
    return Passenger(
        passenger_id=passenger_id,
        pickup=Location(f"{pickup}", pickup, 37.65, 127.05),
        dropoff=Location(f"{dropoff}", dropoff, 37.66, 127.06),
        max_detour_min=detour,
    )


def make_legs(distance_overrides=None, time_overrides=None):
    distance_overrides = distance_overrides or {}
    time_overrides = time_overrides or {}
    stop_ids = {"A", "B", "C", "D"}
    legs = {}
    for origin in stop_ids:
        for destination in stop_ids:
            if origin == destination:
                continue
            legs[(origin, destination)] = Leg(
                distance_overrides.get((origin, destination), 20.0),
                time_overrides.get((origin, destination), 60.0),
            )

    overrides = {
        ("A", "B"): (10.0, 20.0),
        ("C", "D"): (10.0, 20.0),
        ("A", "C"): (1.0, 2.0),
        ("C", "B"): (4.0, 8.0),
        ("B", "D"): (2.0, 4.0),
    }
    for key, (distance, duration) in overrides.items():
        if key not in distance_overrides and key not in time_overrides:
            legs[key] = Leg(distance, duration)
        else:
            legs[key] = Leg(
                distance_overrides.get(key, distance),
                time_overrides.get(key, duration),
            )
    return legs


def two_passenger_problem(max_detour: int = 10, max_capacity: int = 4):
    from algorithms.route_optimizer import RouteProblem

    passengers = [
        make_passenger("A", "A", "B", max_detour),
        make_passenger("B", "C", "D", max_detour),
    ]
    return RouteProblem(
        passengers=passengers,
        legs=make_legs(),
        solo_times_min={"A": 20.0, "B": 20.0},
        max_capacity=max_capacity,
    )
