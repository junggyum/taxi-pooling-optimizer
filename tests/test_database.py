import tempfile
import unittest
from pathlib import Path

from database.db_manager import SimulationStore
from models.passenger import Location, Passenger
from models.route import Leg
from services.fare_service import FareSettings
from services.optimization_service import OptimizationResult
from algorithms.passenger_matching import MatchingResult


class DatabaseTests(unittest.TestCase):
    def test_saves_local_summary_and_loads_result(self):
        passenger = Passenger(
            "P1",
            Location("P1_PU", "출발", 37.65, 127.05),
            Location("P1_DO", "도착", 37.66, 127.06),
            share_allowed=False,
        )
        result = OptimizationResult(
            passengers=(passenger,),
            matching=MatchingResult((), ("P1",), 0, {}),
            individual_legs={"P1": Leg(2.0, 8.0)},
            group_outcomes=(),
            max_capacity=4,
            matching_strategy="efficiency",
            fare_settings=FareSettings(),
        )
        with tempfile.TemporaryDirectory() as folder:
            store = SimulationStore(Path(folder) / "test.sqlite3")
            simulation_id = store.save(result)
            recent = store.recent()
            loaded = store.load_result(simulation_id)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["passenger_count"], 1)
        self.assertEqual(loaded["passengers"][0]["id"], "P1")
        self.assertEqual(loaded["metrics"]["passenger_count"], 1)


if __name__ == "__main__":
    unittest.main()
