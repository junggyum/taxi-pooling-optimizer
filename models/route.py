"""Road-leg and optimized route result models."""

from dataclasses import dataclass, field



@dataclass(frozen=True)
class Leg:
    distance_km: float
    duration_min: float
    geometry: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.distance_km < 0 or self.duration_min < 0:
            raise ValueError("leg distance and duration must not be negative")


@dataclass(frozen=True)
class VisitEvent:
    kind: str
    passenger_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"pickup", "dropoff"}:
            raise ValueError("event kind must be 'pickup' or 'dropoff'")


@dataclass(frozen=True)
class RoutePlan:
    start_passenger_id: str
    events: tuple[VisitEvent, ...]
    total_distance_km: float
    total_duration_min: float
    passenger_ride_minutes: dict[str, float] = field(default_factory=dict)
    detour_minutes: dict[str, float] = field(default_factory=dict)
    detour_violations: dict[str, float] = field(default_factory=dict)
    legs: tuple[Leg, ...] = ()
    feasible: bool = True

    @property
    def geometry(self) -> tuple[tuple[float, float], ...]:
        points: list[tuple[float, float]] = []
        for leg in self.legs:
            for point in leg.geometry:
                if not points or points[-1] != point:
                    points.append(point)
        return tuple(points)
