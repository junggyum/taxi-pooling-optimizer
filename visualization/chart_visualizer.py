"""Plotly figures for rider savings and algorithm comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.optimization_service import OptimizationResult


def passenger_savings_figure(result: OptimizationResult) -> go.Figure:
    rows = result.passenger_summary_rows()
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="승객",
        y=["개별 예상 요금(원)", "합승 부담 요금(원)"],
        barmode="group",
        title="승객별 개별 탑승과 합승 부담 요금 비교",
        labels={"value": "예상 요금(원)", "variable": "요금 유형"},
        color_discrete_sequence=["#94a3b8", "#2563eb"],
    )
    figure.update_layout(legend_title_text="")
    return figure


def algorithm_comparison_figure(result: OptimizationResult) -> go.Figure | None:
    totals: dict[str, dict[str, float | int]] = {}
    for outcome in result.group_outcomes:
        for name, item in outcome.algorithm_results.items():
            values = totals.setdefault(name, {"distance_km": 0.0, "runtime_seconds": 0.0, "solved_groups": 0})
            values["runtime_seconds"] += item.runtime_seconds
            if item.plan is not None:
                values["distance_km"] += item.plan.total_distance_km
                values["solved_groups"] += 1
    rows = [
        {"알고리즘": name, **values}
        for name, values in totals.items()
        if values["solved_groups"] > 0
    ]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    figure = px.bar(
        frame,
        x="알고리즘",
        y="distance_km",
        color="알고리즘",
        hover_data={"runtime_seconds": ":.4f", "solved_groups": True},
        title="합승 그룹별 경로 거리 합계 · 알고리즘 계산 시간은 표로 비교",
        labels={"distance_km": "경로 거리 합계(km)"},
    )
    figure.update_layout(showlegend=False)
    return figure


def algorithm_comparison_rows(result: OptimizationResult) -> list[dict[str, object]]:
    totals: dict[str, dict[str, float | int]] = {}
    for outcome in result.group_outcomes:
        for name, item in outcome.algorithm_results.items():
            row = totals.setdefault(name, {"거리 합계(km)": 0.0, "실행 시간(초)": 0.0, "계산 성공 그룹": 0, "오류": ""})
            row["실행 시간(초)"] = float(row["실행 시간(초)"]) + item.runtime_seconds
            if item.plan:
                row["거리 합계(km)"] = float(row["거리 합계(km)"]) + item.plan.total_distance_km
                row["계산 성공 그룹"] = int(row["계산 성공 그룹"]) + 1
            if item.error:
                row["오류"] = str(item.error)
    return [{"알고리즘": name, **row} for name, row in totals.items()]
