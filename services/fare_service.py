"""Virtual taxi-fare estimates and distance-savings-based pool discounts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from config import DEFAULT_FARE_SETTINGS
from models.passenger import Passenger
from models.route import Leg, RoutePlan


@dataclass(frozen=True)
class FareSettings:
    base_fare_won: int = DEFAULT_FARE_SETTINGS["base_fare_won"]
    included_distance_km: float = DEFAULT_FARE_SETTINGS["included_distance_km"]
    won_per_km: int = DEFAULT_FARE_SETTINGS["won_per_km"]
    won_per_minute: int = DEFAULT_FARE_SETTINGS["won_per_minute"]
    discount_factor: float = DEFAULT_FARE_SETTINGS["discount_factor"]
    max_discount: float = DEFAULT_FARE_SETTINGS["max_discount"]

    def __post_init__(self) -> None:
        if self.base_fare_won < 0 or self.won_per_km < 0 or self.won_per_minute < 0:
            raise ValueError("fare amounts must not be negative")
        if self.included_distance_km < 0:
            raise ValueError("included distance must not be negative")
        if not 0 <= self.discount_factor <= 2:
            raise ValueError("discount_factor must be between 0 and 2")
        if not 0 <= self.max_discount <= 1:
            raise ValueError("max_discount must be between 0 and 1")


@dataclass(frozen=True)
class FareResult:
    individual_fares: dict[str, int]
    passenger_shares: dict[str, int]
    individual_total_won: int
    pooled_charge_total_won: int
    shared_meter_estimate_won: int
    distance_savings_rate: float
    discount_rate: float
    savings_total_won: int


def estimate_fare(distance_km: float, duration_min: float, settings: FareSettings) -> int:
    """Return a rounded virtual fare; this is not Seoul's official meter tariff."""
    extra_distance = max(0.0, distance_km - settings.included_distance_km)
    total = settings.base_fare_won + extra_distance * settings.won_per_km
    total += max(0.0, duration_min) * settings.won_per_minute
    return int(round(total / 100.0) * 100)


def calculate_pool_fares(
    passengers: Sequence[Passenger],
    individual_legs: Mapping[str, Leg],
    pooled_plan: RoutePlan,
    settings: FareSettings | None = None,
) -> FareResult:
    settings = settings or FareSettings()
    fares = {
        p.passenger_id: estimate_fare(
            individual_legs[p.passenger_id].distance_km,
            individual_legs[p.passenger_id].duration_min,
            settings,
        )
        for p in passengers
    }
    individual_total = sum(fares.values())
    individual_distance = sum(individual_legs[p.passenger_id].distance_km for p in passengers)
    if individual_distance <= 0:
        distance_savings = 0.0
    else:
        distance_savings = max(
            0.0,
            (individual_distance - pooled_plan.total_distance_km) / individual_distance,
        )
    discount = min(distance_savings * settings.discount_factor, settings.max_discount)
    pooled_total = int(round(individual_total * (1 - discount) / 100.0) * 100)

    shares: dict[str, int] = {}
    passenger_ids = [p.passenger_id for p in passengers]
    allocated = 0
    for pid in passenger_ids[:-1]:
        share = int(round(pooled_total * fares[pid] / individual_total / 100.0) * 100) if individual_total else 0
        shares[pid] = share
        allocated += share
    if passenger_ids:
        shares[passenger_ids[-1]] = pooled_total - allocated

    return FareResult(
        individual_fares=fares,
        passenger_shares=shares,
        individual_total_won=individual_total,
        pooled_charge_total_won=pooled_total,
        shared_meter_estimate_won=estimate_fare(
            pooled_plan.total_distance_km,
            pooled_plan.total_duration_min,
            settings,
        ),
        distance_savings_rate=distance_savings,
        discount_rate=discount,
        savings_total_won=individual_total - pooled_total,
    )
