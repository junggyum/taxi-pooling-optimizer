"""Folium map rendering for the road-network route results."""

from __future__ import annotations

import folium

from services.optimization_service import OptimizationResult
from services.routing_service import RoadRouter


GROUP_COLORS = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2"]


def create_result_map(router: RoadRouter, result: OptimizationResult) -> folium.Map:
    center = router.boundary.centroid
    map_view = folium.Map(
        location=[center.y, center.x],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.GeoJson(
        router.boundary.__geo_interface__,
        name="노원구 경계",
        style_function=lambda _feature: {
            "color": "#475569", "weight": 2, "fillColor": "#e2e8f0", "fillOpacity": 0.12
        },
    ).add_to(map_view)

    passenger_by_id = {p.passenger_id: p for p in result.passengers}
    for group_index, outcome in enumerate(result.group_outcomes):
        color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
        points = outcome.selected_plan.geometry
        if points:
            folium.PolyLine(
                points,
                color=color,
                weight=6,
                opacity=0.85,
                tooltip=f"합승 {group_index + 1}: {', '.join(outcome.match_group.passenger_ids)}",
            ).add_to(map_view)
        for order, event in enumerate(outcome.selected_plan.events, start=1):
            passenger = passenger_by_id[event.passenger_id]
            location = passenger.pickup if event.kind == "pickup" else passenger.dropoff
            folium.CircleMarker(
                location=[location.latitude, location.longitude],
                radius=7,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"{order}. {event.passenger_id} {'승차' if event.kind == 'pickup' else '하차'} · {location.label}",
            ).add_to(map_view)

    for passenger_id in result.matching.unmatched_passenger_ids:
        leg = result.individual_legs[passenger_id]
        if leg.geometry:
            folium.PolyLine(
                leg.geometry,
                color="#64748b",
                weight=4,
                opacity=0.6,
                dash_array="8 8",
                tooltip=f"{passenger_id}: 개별 이동",
            ).add_to(map_view)
        passenger = passenger_by_id[passenger_id]
        folium.Marker(
            [passenger.pickup.latitude, passenger.pickup.longitude],
            tooltip=f"{passenger_id} 개별 승차 · {passenger.pickup.label}",
            icon=folium.Icon(color="gray", icon="play"),
        ).add_to(map_view)
        folium.Marker(
            [passenger.dropoff.latitude, passenger.dropoff.longitude],
            tooltip=f"{passenger_id} 개별 하차 · {passenger.dropoff.label}",
            icon=folium.Icon(color="gray", icon="stop"),
        ).add_to(map_view)

    folium.LayerControl(collapsed=True).add_to(map_view)
    return map_view
