"""Passenger and stop models used throughout the optimizer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    location_id: str
    label: str
    latitude: float
    longitude: float
    node_id: int | None = None

    def __post_init__(self) -> None:
        if not self.location_id:
            raise ValueError("location_id must not be empty")
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")


@dataclass(frozen=True)
class Passenger:
    passenger_id: str
    pickup: Location
    dropoff: Location
    max_detour_min: int = 10
    share_allowed: bool = True

    def __post_init__(self) -> None:
        if not self.passenger_id:
            raise ValueError("passenger_id must not be empty")
        if not 0 <= self.max_detour_min <= 30:
            raise ValueError("max_detour_min must be between 0 and 30")
        if self.pickup.location_id == self.dropoff.location_id:
            raise ValueError("pickup and dropoff must be different locations")
