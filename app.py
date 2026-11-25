"""Streamlit prototype for taxi-pooling simulation in Nowon-gu, Seoul."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config import (
    DATA_DIR,
    DEFAULT_DETOUR_MINUTES,
    MAX_DEMO_PASSENGERS,
    MIN_DEMO_PASSENGERS,
)
from database.db_manager import SimulationStore
from models.passenger import Passenger
from services.demo_data import generate_demo_passengers
from services.fare_service import FareSettings
from services.optimization_service import OptimizationResult, optimize_passengers
from services.routing_service import MapDataError, RoadRouter
from visualization.chart_visualizer import (
    algorithm_comparison_figure,
    algorithm_comparison_rows,
    passenger_savings_figure,
)
from visualization.map_visualizer import create_result_map


st.set_page_config(page_title="AI 택시 합승 최적화", page_icon="🚕", layout="wide")


@st.cache_resource(show_spinner=False)
def get_router(allow_external_destinations: bool) -> RoadRouter:
    return RoadRouter.load_or_download(allow_external_destinations=allow_external_destinations)


def _sample_data() -> pd.DataFrame:
    sample_path = DATA_DIR / "sample_passengers.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return pd.DataFrame(columns=[
        "passenger_id", "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
        "max_detour_min", "share_allowed",
    ])


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return True
    return str(value).strip().lower() not in {"false", "0", "no", "n", "아니오", "불가"}


def passengers_from_table(
    rows: pd.DataFrame,
    router: RoadRouter,
    input_mode: str,
    allow_external_destinations: bool,
) -> list[Passenger]:
    rows = rows.dropna(how="all").reset_index(drop=True)
    if not MIN_DEMO_PASSENGERS <= len(rows) <= MAX_DEMO_PASSENGERS:
        raise ValueError(f"승객은 {MIN_DEMO_PASSENGERS}~{MAX_DEMO_PASSENGERS}명 입력해 주세요.")

    passengers: list[Passenger] = []
    seen_ids: set[str] = set()
    for index, row in rows.iterrows():
        passenger_id = _clean_text(row.get("passenger_id")) or f"P{index + 1:02d}"
        if passenger_id in seen_ids:
            raise ValueError(f"승객 ID가 중복되었습니다: {passenger_id}")
        seen_ids.add(passenger_id)
        try:
            detour = int(row.get("max_detour_min", DEFAULT_DETOUR_MINUTES))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{passenger_id}: 허용 우회 시간은 0~30분 정수로 입력해 주세요.") from exc
        if not 0 <= detour <= 30:
            raise ValueError(f"{passenger_id}: 허용 우회 시간은 0~30분이어야 합니다.")
        share_allowed = _bool_value(row.get("share_allowed", True))

        if input_mode == "주소 검색":
            pickup_address = _clean_text(row.get("pickup_address"))
            dropoff_address = _clean_text(row.get("dropoff_address"))
            if not pickup_address or not dropoff_address:
                raise ValueError(f"{passenger_id}: 출발지와 목적지 주소를 모두 입력해 주세요.")
            pickup = router.geocode_address(
                pickup_address, f"{passenger_id}_PU", f"{passenger_id} 출발지", require_nowon=True
            )
            dropoff = router.geocode_address(
                dropoff_address,
                f"{passenger_id}_DO",
                f"{passenger_id} 목적지",
                require_nowon=not allow_external_destinations,
            )
        else:
            try:
                coordinates = [
                    float(row[column])
                    for column in ("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon")
                ]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{passenger_id}: 출발·목적 위도와 경도를 모두 입력해 주세요.") from exc
            pickup = router.location_from_coordinates(
                f"{passenger_id}_PU",
                f"{passenger_id} 출발지",
                coordinates[0],
                coordinates[1],
                require_nowon=True,
            )
            dropoff = router.location_from_coordinates(
                f"{passenger_id}_DO",
                f"{passenger_id} 목적지",
                coordinates[2],
                coordinates[3],
                require_nowon=not allow_external_destinations,
            )

        passengers.append(Passenger(
            passenger_id=passenger_id,
            pickup=pickup,
            dropoff=dropoff,
            max_detour_min=detour,
            share_allowed=share_allowed,
        ))
    return passengers


def _render_input_form(input_mode: str, uploaded: pd.DataFrame | None) -> tuple[pd.DataFrame | None, bool, int, int]:
    with st.form("passenger_input"):
        if input_mode == "가상 요청":
            passenger_count = st.slider(
                "가상 승객 수", MIN_DEMO_PASSENGERS, MAX_DEMO_PASSENGERS, 8, step=1
            )
            seed = st.number_input("같은 결과를 만드는 시드", min_value=0, max_value=1_000_000, value=42)
            rows = None
        elif input_mode == "좌표/CSV 입력":
            data = uploaded if uploaded is not None else _sample_data()
            st.caption("표를 직접 수정하거나 아래 CSV 양식을 내려받아 좌표를 넣은 뒤 업로드할 수 있습니다.")
            rows = st.data_editor(
                data,
                key="coordinate_editor",
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "passenger_id": st.column_config.TextColumn("승객 ID"),
                    "pickup_lat": st.column_config.NumberColumn("출발 위도", format="%.6f"),
                    "pickup_lon": st.column_config.NumberColumn("출발 경도", format="%.6f"),
                    "dropoff_lat": st.column_config.NumberColumn("목적 위도", format="%.6f"),
                    "dropoff_lon": st.column_config.NumberColumn("목적 경도", format="%.6f"),
                    "max_detour_min": st.column_config.NumberColumn("허용 우회(분)", min_value=0, max_value=30, step=1),
                    "share_allowed": st.column_config.CheckboxColumn("합승 가능"),
                },
            )
            passenger_count = 0
            seed = 42
        else:
            data = pd.DataFrame({
                "passenger_id": [f"P{i:02d}" for i in range(1, 6)],
                "pickup_address": [""] * 5,
                "dropoff_address": [""] * 5,
                "max_detour_min": [DEFAULT_DETOUR_MINUTES] * 5,
                "share_allowed": [True] * 5,
            })
            st.caption("주소 검색어는 좌표 변환을 위해 OpenStreetMap 지오코더로 전송됩니다. 이름·전화번호 등은 입력하지 마세요.")
            rows = st.data_editor(
                data,
                key="address_editor",
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "passenger_id": st.column_config.TextColumn("승객 ID"),
                    "pickup_address": st.column_config.TextColumn("노원구 출발지 주소"),
                    "dropoff_address": st.column_config.TextColumn("목적지 주소"),
                    "max_detour_min": st.column_config.NumberColumn("허용 우회(분)", min_value=0, max_value=30, step=1),
                    "share_allowed": st.column_config.CheckboxColumn("합승 가능"),
                },
            )
            passenger_count = 0
            seed = 42
        submitted = st.form_submit_button("🚕 경로와 합승 최적화", type="primary", width="stretch")
    return rows, submitted, int(passenger_count), int(seed)


def _render_results(result: OptimizationResult, router: RoadRouter) -> None:
    metrics = result.summary_metrics()
    first_row = st.columns(4)
    first_row[0].metric("승객", f"{metrics['passenger_count']}명")
    first_row[1].metric("합승 매칭", f"{metrics['pooled_passenger_count']}명")
    first_row[2].metric("매칭 성공률", f"{metrics['matching_success_rate']:.1%}")
    first_row[3].metric("허용 우회 준수율", f"{metrics['max_detour_compliance_rate']:.1%}" if metrics['max_detour_compliance_rate'] is not None else "해당 없음")
    second_row = st.columns(4)
    second_row[0].metric("개별 주행 거리 합", f"{metrics['individual_distance_km']:.2f} km")
    second_row[1].metric("계산된 주행 거리 합", f"{metrics['pooled_distance_km']:.2f} km", f"절감 {metrics['distance_saved_km']:.2f} km · {metrics['distance_savings_rate']:.1%}")
    second_row[2].metric("개별 승객 시간 합", f"{metrics['individual_time_min']:.1f}분")
    second_row[3].metric("합승 승객 시간 합", f"{metrics['pooled_passenger_time_min']:.1f}분")
    third_row = st.columns(3)
    third_row[0].metric("개별 예상 요금 합", f"{metrics['individual_fare_won']:,}원")
    third_row[1].metric("합승 예상 부담 합", f"{metrics['pooled_charge_won']:,}원")
    third_row[2].metric("예상 요금 절감", f"{metrics['fare_saved_won']:,}원", f"{metrics['fare_savings_rate']:.1%}")
    st.caption(f"택시 차량 운행시간 합: {metrics['pooled_vehicle_operating_time_min']:.1f}분")

    if not result.group_outcomes:
        st.info("조건을 만족하면서 거리 절감 효과가 있는 합승 조합을 찾지 못했습니다. 승객은 개별 탑승으로 계산했습니다.")
    st.subheader("노원구 도로 경로")
    st.caption("도로별 가정 평균 속도를 사용합니다. 실시간 교통정보와 공식 택시 미터 요금은 반영하지 않습니다.")
    route_map = create_result_map(router, result)
    st_folium(route_map, height=590, use_container_width=True, returned_objects=[])

    st.subheader("승객별 시간 및 요금")
    st.dataframe(pd.DataFrame(result.passenger_summary_rows()), width="stretch", hide_index=True)
    st.plotly_chart(passenger_savings_figure(result), width="stretch")

    st.subheader("합승 그룹 및 승·하차 순서")
    if result.group_outcomes:
        for index, outcome in enumerate(result.group_outcomes, start=1):
            with st.expander(f"합승 {index}: {', '.join(outcome.match_group.passenger_ids)}", expanded=True):
                order_text = []
                passenger_by_id = {p.passenger_id: p for p in outcome.match_group.passengers}
                for stop_number, event in enumerate(outcome.selected_plan.events, start=1):
                    person = passenger_by_id[event.passenger_id]
                    location = person.pickup if event.kind == "pickup" else person.dropoff
                    order_text.append(f"{stop_number}. {event.passenger_id} {'승차' if event.kind == 'pickup' else '하차'} · {location.label}")
                st.write("\n".join(order_text))
                st.caption(
                    f"총 거리 {outcome.selected_plan.total_distance_km:.2f} km · "
                    f"차량 이동 {outcome.selected_plan.total_duration_min:.1f}분 · "
                    f"합승 할인 {outcome.fare_result.discount_rate:.1%}"
                )

    st.subheader("알고리즘 비교")
    comparison = algorithm_comparison_rows(result)
    if comparison:
        figure = algorithm_comparison_figure(result)
        if figure is not None:
            st.plotly_chart(figure, width="stretch")
        st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
        errors = [f"{row['알고리즘']}: {row['오류']}" for row in comparison if row["오류"]]
        if errors:
            st.warning("일부 비교 알고리즘이 결과를 내지 못했습니다: " + " · ".join(errors))
    else:
        st.info("비교할 합승 그룹이 없습니다.")


st.title("🚕 AI 택시 합승 매칭 및 경로 최적화")
st.caption("서울 노원구 OSM 도로망 기반 시뮬레이션 · 실시간 교통과 공식 택시 요금은 반영하지 않습니다.")

with st.sidebar:
    st.header("시뮬레이션 설정")
    input_mode = st.radio("승객 데이터", ["가상 요청", "좌표/CSV 입력", "주소 검색"])
    max_capacity = st.slider("택시 최대 탑승 인원", min_value=2, max_value=4, value=4)
    matching_labels = {
        "거리·우회 효율": "efficiency",
        "가까운 출발지": "nearest",
        "경로 방향 유사도": "similarity",
    }
    strategy_label = st.selectbox("매칭 방식", list(matching_labels))
    allow_external_destinations = st.checkbox("노원구 밖 목적지 허용 (추가 도로망 필요)", value=False)
    discount_factor = st.slider("거리 절감률 반영 배수", 0.0, 2.0, 1.0, step=0.1)
    max_discount = st.slider("합승 할인 상한", 0, 50, 30, step=5) / 100

st.info(
    "개발용 프로토타입입니다. 지도는 OpenStreetMap 도로 자료를 사용하고, 이동 시간은 도로 유형별 평균 속도로 추정합니다. "
    "이전 실행 내역은 이 컴퓨터의 SQLite 파일에만 저장됩니다."
)
st.caption("지도 자료: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright) · OpenStreetMap 데이터는 ODbL로 제공됩니다.")

uploaded_frame: pd.DataFrame | None = None
if input_mode == "좌표/CSV 입력":
    upload = st.file_uploader("CSV 승객 파일", type=["csv"])
    if upload is not None:
        try:
            uploaded_frame = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"CSV를 읽지 못했습니다: {exc}")

rows, submitted, passenger_count, seed = _render_input_form(input_mode, uploaded_frame)

if submitted:
    try:
        with st.spinner("노원구 도로망을 불러오고 승객 경로를 계산하고 있습니다…"):
            router = get_router(allow_external_destinations)
            if input_mode == "가상 요청":
                passengers = generate_demo_passengers(router, int(passenger_count), int(seed))
            else:
                assert rows is not None
                passengers = passengers_from_table(
                    rows, router, input_mode, allow_external_destinations
                )
            settings = FareSettings(discount_factor=discount_factor, max_discount=max_discount)
            result = optimize_passengers(
                passengers,
                router,
                max_capacity=max_capacity,
                matching_strategy=matching_labels[strategy_label],
                fare_settings=settings,
            )
        st.session_state["last_result"] = result
        st.session_state["last_external"] = allow_external_destinations
        try:
            simulation_id = SimulationStore().save(result)
            st.session_state["last_simulation_id"] = simulation_id
        except Exception as exc:
            st.warning(f"최적화는 끝났지만 로컬 실행 내역을 저장하지 못했습니다: {exc}")
    except MapDataError as exc:
        st.error(f"도로망 또는 주소 데이터를 불러오지 못했습니다. {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"계산 중 문제가 발생했습니다: {type(exc).__name__}: {exc}")


results_tab, history_tab = st.tabs(["합승 결과", "이전 실행"])
with results_tab:
    current_result = st.session_state.get("last_result")
    if current_result is None:
        st.markdown("좌측에서 입력 방식을 고르고 **경로와 합승 최적화**를 누르면 결과를 확인할 수 있습니다.")
    else:
        result_router = get_router(bool(st.session_state.get("last_external", False)))
        _render_results(current_result, result_router)
        if st.session_state.get("last_simulation_id"):
            st.caption(f"이번 결과는 이 컴퓨터의 실행 내역 #{st.session_state['last_simulation_id']}에 저장했습니다.")

with history_tab:
    st.subheader("이 컴퓨터의 최근 실행 내역")
    st.caption("좌표와 결과는 프로젝트 폴더의 data/simulations.sqlite3에 저장되며 자동으로 외부에 전송하지 않습니다.")
    try:
        history = SimulationStore().recent(limit=30)
        if history:
            frame = pd.DataFrame(history).rename(columns={
                "id": "실행 번호", "created_at": "저장 시각(UTC)", "passenger_count": "승객 수",
                "pooled_passenger_count": "합승 승객 수", "distance_saved_km": "거리 절감(km)",
                "fare_saved_won": "예상 요금 절감(원)", "matching_strategy": "매칭 방식",
                "max_capacity": "최대 탑승 인원",
            })
            st.dataframe(frame, width="stretch", hide_index=True)
        else:
            st.info("저장된 실행 내역이 없습니다.")
    except Exception as exc:
        st.error(f"이전 실행 내역을 읽지 못했습니다: {exc}")
