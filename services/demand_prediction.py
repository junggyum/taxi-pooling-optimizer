"""Demo-only, synthetic taxi-demand forecast independent of route optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


DEMO_ZONES = {
    0: "노원역 주변",
    1: "상계역 주변",
    2: "공릉역·대학가",
    3: "하계역·병원가",
    4: "월계역 주변",
}


@dataclass(frozen=True)
class DemandForecast:
    zone_code: int
    zone_name: str
    hour: int
    weekday: int
    expected_requests: float
    label: str = "가상 데이터 기반 시연값"


def _synthetic_count(hour: int, weekday: int, zone: int, rng: np.random.Generator) -> int:
    rush = 4.0 * math.exp(-((hour - 8) / 2.1) ** 2) + 5.0 * math.exp(-((hour - 18) / 2.5) ** 2)
    evening = 2.0 * math.exp(-((hour - 22) / 2.0) ** 2)
    weekend = 1.5 if weekday >= 5 else 0.0
    zone_effect = [1.5, 1.0, 1.2, 0.8, 0.7][zone]
    mean = max(0.3, zone_effect + rush + evening + weekend)
    return int(max(0, rng.poisson(mean)))


def synthetic_training_data(samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Create clearly synthetic rows for a demo; these are not measured taxi calls."""
    if samples < 100:
        raise ValueError("at least 100 synthetic rows are required")
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 24, size=samples)
    weekdays = rng.integers(0, 7, size=samples)
    zones = rng.integers(0, len(DEMO_ZONES), size=samples)
    counts = [
        _synthetic_count(int(hour), int(day), int(zone), rng)
        for hour, day, zone in zip(hours, weekdays, zones)
    ]
    return pd.DataFrame({"hour": hours, "weekday": weekdays, "zone_code": zones, "requests": counts})


def predict_demo_demand(
    hour: int,
    weekday: int,
    zone_code: int,
    seed: int = 42,
) -> DemandForecast:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= weekday <= 6:
        raise ValueError("weekday must be between 0 and 6")
    if zone_code not in DEMO_ZONES:
        raise ValueError(f"unknown zone code: {zone_code}")
    training = synthetic_training_data(seed=seed)
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, random_state=seed, n_jobs=1)
    model.fit(training[["hour", "weekday", "zone_code"]], training["requests"])
    estimate = float(model.predict(pd.DataFrame({
        "hour": [hour], "weekday": [weekday], "zone_code": [zone_code]
    }))[0])
    return DemandForecast(zone_code, DEMO_ZONES[zone_code], hour, weekday, max(0.0, estimate))
