"""Local-only SQLite persistence for simulation summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Any

from config import DATABASE_PATH
from services.optimization_service import OptimizationResult


class SimulationStore:
    """Store simulations on this device; no network calls are made here."""

    def __init__(self, path: str | Path = DATABASE_PATH) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    passenger_count INTEGER NOT NULL,
                    pooled_passenger_count INTEGER NOT NULL,
                    distance_saved_km REAL NOT NULL,
                    fare_saved_won INTEGER NOT NULL,
                    matching_strategy TEXT NOT NULL,
                    max_capacity INTEGER NOT NULL,
                    result_json TEXT NOT NULL
                )"""
            )

    @staticmethod
    def _result_payload(result: OptimizationResult) -> dict[str, Any]:
        passengers = [
            {
                "id": p.passenger_id,
                "pickup": {"label": p.pickup.label, "lat": p.pickup.latitude, "lon": p.pickup.longitude},
                "dropoff": {"label": p.dropoff.label, "lat": p.dropoff.latitude, "lon": p.dropoff.longitude},
                "max_detour_min": p.max_detour_min,
                "requested_at": p.requested_at,
                "share_allowed": p.share_allowed,
            }
            for p in result.passengers
        ]
        groups = []
        for outcome in result.group_outcomes:
            groups.append({
                "passenger_ids": list(outcome.match_group.passenger_ids),
                "distance_km": outcome.selected_plan.total_distance_km,
                "duration_min": outcome.selected_plan.total_duration_min,
                "events": [
                    {"kind": event.kind, "passenger_id": event.passenger_id}
                    for event in outcome.selected_plan.events
                ],
                "detour_min": outcome.selected_plan.detour_minutes,
                "fare_shares_won": outcome.fare_result.passenger_shares,
                "algorithms": {
                    name: {
                        "distance_km": item.plan.total_distance_km if item.plan else None,
                        "runtime_seconds": item.runtime_seconds,
                        "error": item.error,
                    }
                    for name, item in outcome.algorithm_results.items()
                },
            })
        return {
            "passengers": passengers,
            "groups": groups,
            "metrics": result.summary_metrics(),
        }

    def save(self, result: OptimizationResult) -> int:
        self.initialize()
        metrics = result.summary_metrics()
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO simulations (
                    created_at, passenger_count, pooled_passenger_count,
                    distance_saved_km, fare_saved_won, matching_strategy,
                    max_capacity, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    created_at,
                    len(result.passengers),
                    int(metrics["pooled_passenger_count"]),
                    float(metrics["distance_saved_km"]),
                    int(metrics["fare_saved_won"]),
                    result.matching_strategy,
                    result.max_capacity,
                    json.dumps(self._result_payload(result), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, created_at, passenger_count, pooled_passenger_count,
                          distance_saved_km, fare_saved_won, matching_strategy, max_capacity
                   FROM simulations ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def load_result(self, simulation_id: int) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM simulations WHERE id = ?", (simulation_id,)
            ).fetchone()
        return json.loads(row["result_json"]) if row else None
