"""Project settings shared by the UI and service modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ROAD_NETWORK_DIR = DATA_DIR / "road_network"
ROAD_GRAPH_PATH = ROAD_NETWORK_DIR / "nowon_drive.graphml"
OSM_CACHE_DIR = ROAD_NETWORK_DIR / "http"
DATABASE_PATH = DATA_DIR / "simulations.sqlite3"

NOWON_PLACE_QUERY = "Nowon-gu, Seoul, South Korea"
OSM_HTTP_USER_AGENT = (
    "AI-Taxi-Pooling-Optimizer/0.1 "
    "(+https://github.com/junggyum/taxi-pooling-optimizer)"
)
DEFAULT_MAX_CAPACITY = 4
MIN_DETOUR_MINUTES = 0
MAX_DETOUR_MINUTES = 30
DEFAULT_DETOUR_MINUTES = 10
MIN_DEMO_PASSENGERS = 5
MAX_DEMO_PASSENGERS = 20

# These are demonstration assumptions, not Seoul's official taxi-meter tariff.
DEFAULT_FARE_SETTINGS = {
    "base_fare_won": 4_800,
    "included_distance_km": 1.6,
    "won_per_km": 1_000,
    "won_per_minute": 80,
    "discount_factor": 1.0,
    "max_discount": 0.30,
}

ROAD_TYPE_SPEEDS_KPH = {
    "motorway": 70,
    "trunk": 50,
    "primary": 40,
    "secondary": 35,
    "tertiary": 30,
    "residential": 25,
    "service": 15,
}
FALLBACK_SPEED_KPH = 25
