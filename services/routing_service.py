"""OpenStreetMap road graph loading, geocoding, and directed road routing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import osmnx as ox
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from config import (
    FALLBACK_SPEED_KPH,
    NOWON_PLACE_QUERY,
    OSM_HTTP_USER_AGENT,
    OSM_CACHE_DIR,
    ROAD_NETWORK_DIR,
    ROAD_TYPE_SPEEDS_KPH,
    ROAD_GRAPH_PATH,
)
from models.passenger import Location, Passenger
from models.route import Leg


class MapDataError(RuntimeError):
    """Raised when a map query or road-network operation cannot be completed."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2) ** 2
    value += math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(value))


class RoadRouter:
    """Route on a cached OSMnx driving graph and keep all estimates labeled as estimates."""

    def __init__(self, graph: nx.MultiDiGraph, boundary, graph_path: Path, allow_external_destinations: bool = False):
        self.graph = graph
        self.boundary = boundary
        self.graph_path = Path(graph_path)
        self.allow_external_destinations = allow_external_destinations
        self._leg_cache: dict[tuple[int, int], Leg] = {}
        self._node_coordinates = [
            (node, float(data["y"]), float(data["x"]))
            for node, data in graph.nodes(data=True)
            if "x" in data and "y" in data
        ]
        if not self._node_coordinates:
            raise MapDataError("OSM graph has no latitude/longitude nodes")

    @classmethod
    def load_or_download(
        cls,
        graph_path: str | Path | None = None,
        allow_external_destinations: bool = False,
    ) -> "RoadRouter":
        """Load the cached Nowon graph or download it once from OpenStreetMap."""
        graph_file = Path(graph_path or ROAD_GRAPH_PATH)
        if allow_external_destinations and graph_path is None:
            graph_file = ROAD_NETWORK_DIR / "nowon_drive_external_buffer.graphml"

        try:
            graph_file.parent.mkdir(parents=True, exist_ok=True)
            OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ox.settings.use_cache = True
            ox.settings.cache_folder = str(OSM_CACHE_DIR)
            ox.settings.http_user_agent = OSM_HTTP_USER_AGENT
            boundary = cls._load_boundary()

            if graph_file.exists():
                graph = ox.load_graphml(graph_file)
            elif allow_external_destinations:
                boundary_gdf = ox.geocode_to_gdf(NOWON_PLACE_QUERY)
                buffered = boundary_gdf.to_crs(epsg=5186).geometry.buffer(3_000).to_crs(epsg=4326)
                graph = ox.graph_from_polygon(buffered.unary_union, network_type="drive", simplify=True)
                ox.save_graphml(graph, filepath=graph_file)
            else:
                graph = ox.graph_from_place(NOWON_PLACE_QUERY, network_type="drive", simplify=True)
                ox.save_graphml(graph, filepath=graph_file)

            graph = cls._ensure_travel_times(graph)
            # Keep speeds/travel times in the cache so later launches can work offline.
            ox.save_graphml(graph, filepath=graph_file)
            return cls(graph, boundary, graph_file, allow_external_destinations)
        except Exception as exc:
            raise MapDataError(
                "노원구 OSM 도로망을 불러오지 못했습니다. 인터넷 연결과 OSM 응답을 확인한 뒤 다시 시도해 주세요."
            ) from exc

    @staticmethod
    def _ensure_travel_times(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        has_speeds = all("speed_kph" in data for *_, data in graph.edges(data=True))
        has_times = all("travel_time" in data for *_, data in graph.edges(data=True))
        if not has_speeds:
            graph = ox.add_edge_speeds(
                graph,
                hwy_speeds=ROAD_TYPE_SPEEDS_KPH,
                fallback=FALLBACK_SPEED_KPH,
            )
        if not has_times:
            graph = ox.add_edge_travel_times(graph)
        return graph

    @staticmethod
    def _load_boundary():
        boundary_path = ROAD_NETWORK_DIR / "nowon_boundary.geojson"
        if boundary_path.exists():
            payload = json.loads(boundary_path.read_text(encoding="utf-8"))
            feature = payload["features"][0]
            return shape(feature["geometry"])

        ROAD_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
        boundary_gdf = ox.geocode_to_gdf(NOWON_PLACE_QUERY)
        geometry = unary_union(list(boundary_gdf.geometry))
        payload = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"source": "OpenStreetMap"}, "geometry": mapping(geometry)}],
        }
        boundary_path.write_text(json.dumps(payload), encoding="utf-8")
        return geometry

    def is_in_nowon(self, latitude: float, longitude: float) -> bool:
        return bool(self.boundary.covers(Point(longitude, latitude)))

    def nearest_node(self, latitude: float, longitude: float) -> int:
        # A local linear scan is fast for this district-sized graph and needs no SciPy.
        return min(
            self._node_coordinates,
            key=lambda item: haversine_km(latitude, longitude, item[1], item[2]),
        )[0]

    def location_from_coordinates(
        self,
        location_id: str,
        label: str,
        latitude: float,
        longitude: float,
        require_nowon: bool = True,
    ) -> Location:
        if require_nowon and not self.is_in_nowon(latitude, longitude):
            raise ValueError("출발지는 서울 노원구 안의 주소를 입력해 주세요.")
        node_id = self.nearest_node(latitude, longitude)
        node_data = self.graph.nodes[node_id]
        distance_to_road = haversine_km(latitude, longitude, float(node_data["y"]), float(node_data["x"]))
        if distance_to_road > 5.0:
            raise ValueError("입력 위치에서 가까운 차량 통행 도로를 찾지 못했습니다.")
        if not require_nowon and not self.allow_external_destinations and not self.is_in_nowon(latitude, longitude):
            raise ValueError("노원구 외 목적지는 외부 목적지 허용 설정을 켜야 합니다.")
        return Location(location_id, label, latitude, longitude, int(node_id))

    def geocode_address(
        self,
        address: str,
        location_id: str,
        label: str,
        require_nowon: bool = True,
    ) -> Location:
        try:
            result = ox.geocode(address)
        except Exception as exc:
            raise MapDataError(f"주소를 찾지 못했습니다: {address}") from exc
        if not result:
            raise MapDataError(f"주소 검색 결과가 없습니다: {address}")
        latitude, longitude = result
        return self.location_from_coordinates(
            location_id,
            label,
            float(latitude),
            float(longitude),
            require_nowon=require_nowon,
        )

    def location_for_node(self, node_id: int, label: str | None = None) -> Location:
        data = self.graph.nodes[node_id]
        return Location(
            location_id=str(node_id),
            label=label or f"노원구 가상 지점 {node_id}",
            latitude=float(data["y"]),
            longitude=float(data["x"]),
            node_id=int(node_id),
        )

    @staticmethod
    def _best_edge_data(graph: nx.MultiDiGraph, origin: int, destination: int) -> Mapping:
        edge_map = graph.get_edge_data(origin, destination)
        if not edge_map:
            raise MapDataError(f"도로 구간을 찾지 못했습니다: {origin} -> {destination}")
        return min(edge_map.values(), key=lambda data: float(data.get("length", math.inf)))

    def _make_leg_from_path(self, path: Sequence[int]) -> Leg:
        if len(path) < 2:
            return Leg(0.0, 0.0, ((float(self.graph.nodes[path[0]]["y"]), float(self.graph.nodes[path[0]]["x"])),), tuple(path))

        distance_m = 0.0
        duration_seconds = 0.0
        geometry: list[tuple[float, float]] = []
        first_data = self.graph.nodes[path[0]]
        geometry.append((float(first_data["y"]), float(first_data["x"])))

        for origin, destination in zip(path, path[1:]):
            data = self._best_edge_data(self.graph, origin, destination)
            distance_m += float(data.get("length", 0.0))
            if "travel_time" in data:
                duration_seconds += float(data["travel_time"])
            else:
                speed = float(data.get("speed_kph", FALLBACK_SPEED_KPH))
                duration_seconds += float(data.get("length", 0.0)) / max(speed, 1.0) * 3.6

            edge_geometry = data.get("geometry")
            if edge_geometry is not None and hasattr(edge_geometry, "coords"):
                edge_points = [(float(lat), float(lon)) for lon, lat in edge_geometry.coords]
            else:
                node = self.graph.nodes[destination]
                edge_points = [(float(node["y"]), float(node["x"]))]
            for point in edge_points:
                if geometry[-1] != point:
                    geometry.append(point)

        return Leg(
            distance_km=distance_m / 1000,
            duration_min=duration_seconds / 60,
            geometry=tuple(geometry),
        )

    def route_between(self, origin: Location, destination: Location) -> Leg:
        if origin.node_id is None or destination.node_id is None:
            raise ValueError("both locations must be snapped to road nodes")
        key = (int(origin.node_id), int(destination.node_id))
        if key in self._leg_cache:
            return self._leg_cache[key]
        if key[0] == key[1]:
            leg = Leg(
                0.0,
                0.0,
                ((origin.latitude, origin.longitude), (destination.latitude, destination.longitude)),
            )
            self._leg_cache[key] = leg
            return leg
        try:
            path = nx.shortest_path(self.graph, key[0], key[1], weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise MapDataError(f"두 위치 사이의 차량 도로 경로를 찾지 못했습니다: {origin.label} → {destination.label}") from exc
        leg = self._make_leg_from_path(path)
        self._leg_cache[key] = leg
        return leg

    def build_leg_matrix(self, passengers: Sequence[Passenger]) -> dict[tuple[str, str], Leg]:
        """Build all reachable directed stop-to-stop legs with one Dijkstra run per road node."""
        locations: dict[str, Location] = {}
        for passenger in passengers:
            locations[passenger.pickup.location_id] = passenger.pickup
            locations[passenger.dropoff.location_id] = passenger.dropoff

        by_node: dict[int, list[Location]] = {}
        for location in locations.values():
            if location.node_id is None:
                raise ValueError(f"location {location.location_id} has no road node")
            by_node.setdefault(int(location.node_id), []).append(location)

        matrix: dict[tuple[str, str], Leg] = {}
        for source_node, origin_locations in by_node.items():
            try:
                _, paths = nx.single_source_dijkstra(self.graph, source_node, weight="length")
            except nx.NodeNotFound as exc:
                raise MapDataError(f"road node {source_node} is missing from the downloaded graph") from exc
            for destination_node, destination_locations in by_node.items():
                path = paths.get(destination_node)
                if path is None:
                    continue
                leg = self._make_leg_from_path(path)
                self._leg_cache[(source_node, destination_node)] = leg
                for origin in origin_locations:
                    for destination in destination_locations:
                        if origin.location_id == destination.location_id:
                            continue
                        matrix[(origin.location_id, destination.location_id)] = leg
        for location_id in locations:
            matrix[(location_id, location_id)] = Leg(0.0, 0.0)
        return matrix


def individual_route_legs(
    passengers: Iterable[Passenger],
    matrix: Mapping[tuple[str, str], Leg],
) -> dict[str, Leg]:
    result: dict[str, Leg] = {}
    for passenger in passengers:
        key = (passenger.pickup.location_id, passenger.dropoff.location_id)
        if key not in matrix:
            raise MapDataError(
                f"개별 도로 경로를 찾지 못했습니다: {passenger.pickup.label} → {passenger.dropoff.label}"
            )
        result[passenger.passenger_id] = matrix[key]
    return result
