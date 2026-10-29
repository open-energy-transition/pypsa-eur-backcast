# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the backcast dashboard data layer, extract and app."""

import shutil
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import pytest
from shapely.geometry import box

from dashboard import data
from dashboard.extract import extract
from dashboard.narrative import HISTORY, NAMES, Step
from dashboard.views import METHOD_PAGE, PAGES, STORY_PAGE, ZONE_PAGE
from scripts.lib.backcast_data import (
    NO_MARGINAL,
    area_capacities,
    marginal_carriers,
    monthly_bias,
    monthly_mae,
    price_metrics,
    stacked_generation,
    top_errors,
)
from scripts.lib.backcast_plotting import ZONE_ORDER

HOURS = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
AREA_TO_BUS = pd.Series({"DE00": "DE_LU", "FR00": "FR"}, name="entsoe_area")
FILES = {
    "prices",
    "load",
    "generation",
    "flows",
    "marginal",
    "metrics",
    "monthly_bias",
}


def solved_network(hours: pd.DatetimeIndex = HOURS) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(hours)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    n.add("Link", "DE00->FR00", bus0="DE00", bus1="FR00", carrier="DC", p_nom=1.0)
    n.add(
        "Generator",
        ["DE00 CCGT", "DE00 solar", "FR00 nuclear", "FR00 load"],
        bus=["DE00", "DE00", "FR00", "FR00"],
        carrier=["CCGT", "solar", "nuclear", "load"],
        marginal_cost=[50.0, 0.0, 10.0, 1000.0],
    )
    n.add("Load", ["DE00 load", "FR00 load"], bus=["DE00", "FR00"])
    n.add("StorageUnit", "FR00 hydro", bus="FR00", carrier="hydro")
    day = np.tile(np.r_[np.full(12, 50.0), np.full(12, 10.5)], len(hours) // 24)
    n.buses_t.marginal_price = pd.DataFrame({"DE00": day, "FR00": 10.0}, hours)
    n.generators_t.p = pd.DataFrame(
        {"DE00 CCGT": 1.0, "DE00 solar": 2.0, "FR00 nuclear": 3.0, "FR00 load": 0.0},
        hours,
    )
    n.links_t.p0 = pd.DataFrame({"DE00->FR00": 1.0}, hours)
    n.loads_t.p = pd.DataFrame({"DE00 load": 3.0, "FR00 load": 2.0}, hours)
    n.storage_units_t.p = pd.DataFrame({"FR00 hydro": 1.0}, hours)
    n.storage_units_t.state_of_charge = pd.DataFrame({"FR00 hydro": 5.0}, hours)
    n.storage_units_t.inflow = pd.DataFrame({"FR00 hydro": 0.5}, hours)
    return n


def write_run(tmp: Path, name: str, hours: pd.DatetimeIndex) -> Path:
    n = solved_network(hours)
    observed = n.buses_t.marginal_price + np.sin(np.arange(len(hours)))[:, None]
    model = n.buses_t.marginal_price
    entsoe = pd.DataFrame(
        {("DE_LU", "gas"): 1.0, ("FR", "nuclear"): 3.0}, hours
    ).rename_axis(columns=["area", "carrier"])
    paths = {
        "network": tmp / "network.nc",
        "prices": tmp / "prices.csv",
        "generation": tmp / "generation.csv",
        "exchanges": tmp / "exchanges.csv",
        "capacities": tmp / "capacities.csv",
        "regions": tmp / "regions.geojson",
        "metrics": tmp / "metrics.csv",
        "monthly_bias": tmp / "monthly_bias.csv",
        "area_to_bus": tmp / "area_to_bus.csv",
    }
    n.export_to_netcdf(paths["network"])
    observed.to_csv(paths["prices"])
    entsoe.to_csv(paths["generation"])
    pd.DataFrame({"DE_LU->FR": 2.0, "FR->DE_LU": 1.0}, hours).to_csv(paths["exchanges"])
    pd.DataFrame({"DE00->FR00": 5.0, "FR00->DE00": 4.0}, hours).to_csv(
        paths["capacities"]
    )
    regions = gpd.GeoDataFrame(
        {"name": AREA_TO_BUS.index}, geometry=[box(6, 48, 14, 54), box(-4, 43, 6, 50)]
    )
    regions.set_crs(4326).to_file(paths["regions"], driver="GeoJSON")
    price_metrics(model, observed).to_csv(paths["metrics"])
    monthly_bias(model, observed).to_csv(paths["monthly_bias"])
    AREA_TO_BUS.rename_axis("bus").to_csv(paths["area_to_bus"])
    out = tmp / "cache" / name
    extract(paths, out, name)
    return out


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return write_run(tmp_path_factory.mktemp("run"), "test-run", HOURS)


@pytest.fixture(scope="module")
def validation_cache(tmp_path_factory, run_dir) -> Path:
    tmp = tmp_path_factory.mktemp("validation")
    write_run(
        tmp, "test-run-2019", pd.date_range("2019-01-01", periods=len(HOURS), freq="h")
    )
    shutil.copytree(run_dir, tmp / "cache" / run_dir.name)
    return tmp / "cache"


def test_extract_writes_files_and_marginal_carriers(run_dir):
    written = {p.stem for p in run_dir.glob("*.parquet")}
    run = data.load_run(run_dir)

    assert written == FILES
    assert run.model.shape == (len(HOURS), 2)
    assert run.buses.to_dict() == {"DE_LU": "DE00", "FR": "FR00"}
    assert run.flows["ntc"].columns.tolist() == ["DE_LU->FR", "FR->DE_LU"]
    assert run.marginal["DE00"].iloc[:13].tolist() == [*["gas"] * 12, "nuclear"]
    assert set(run.marginal["FR00"]) == {"nuclear"}
    assert run.year == 2023 and data.stale_sources(run.manifest) == []


def test_fetch_cache_unpacks_runs_and_marks_source(run_dir, tmp_path):
    archive = tmp_path / "dashboard-cache.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    cache = tmp_path / "cache"
    data.fetch_cache(cache, archive.as_uri())

    assert data.discover_runs(cache) == [run_dir.name]
    assert (cache / data.FETCH_MARKER).read_text() == archive.as_uri()


@pytest.mark.parametrize(
    "price, expected",
    [(50.4, "gas"), (10.0, "nuclear"), (0.0, "solar"), (30.0, NO_MARGINAL)],
)
def test_marginal_carriers_tolerance(price, expected):
    n = solved_network()
    n.buses_t.marginal_price["DE00"] = price

    assert set(marginal_carriers(n)["DE00"]) == {expected}


def test_area_capacities_sum_directed_area_pairs():
    capacities = pd.DataFrame(
        {"DE00->FR00": [1.0], "DE01->FR00": [2.0], "DE00->DE01": [9.0]}
    )
    area_to_bus = pd.Series({"DE00": "DE_LU", "DE01": "DE_LU", "FR00": "FR"})

    result = area_capacities(capacities, area_to_bus)

    assert result.to_dict("list") == {"DE_LU->FR": [3.0]}


def test_monthly_mae_and_top_errors():
    index = pd.date_range("2023-01-31", periods=4, freq="12h")
    error = pd.DataFrame({"A": [1.0, -3, 2, -2]}, index)
    observed = pd.DataFrame({"A": [1.0, 2, 3, 4]}, index)

    assert monthly_mae(error)["A"].tolist() == [2.0, 2.0]
    assert top_errors(error, observed, 0.25)["A"] == -2.0


def test_stacked_generation_fills_carriers_in_gw():
    generation = pd.DataFrame({"gas": [1000.0, 2000.0]}, HOURS[:2])

    stack = stacked_generation(generation, HOURS[:3])

    assert stack.shape == (3, len(stack.columns)) and stack.gas.tolist() == [1, 2, 0]


@pytest.mark.parametrize(
    "preset, kwargs, hours",
    [
        ("year", {}, len(HOURS)),
        ("February", {}, 28 * 24),
        ("winter week", {}, 168),
        ("custom", {"dates": (pd.Timestamp("2023-03-01"),) * 2}, 24),
    ],
)
@pytest.mark.parametrize("aggregation, factor", [("hourly", 1), ("daily", 24)])
def test_period_index_and_aggregate(preset, kwargs, hours, aggregation, factor):
    weeks = {"winter": HOURS[:168], "summer": HOURS[-168:]}
    index = data.period_index(HOURS, preset, weeks, **kwargs)
    prices = pd.Series(1.0, index)

    assert len(index) == hours
    assert len(data.aggregate(prices, aggregation)) == hours // factor


@pytest.mark.parametrize(
    "errors, expected",
    [
        ([0, 1, 5, 6, 4, 2, 0], (2, 4, 5.0)),
        ([-1, -8, -2, 3, 0, 0, 0], (1, 1, -8.0)),
    ],
)
def test_worst_hours_window(errors, expected):
    assert data.worst_hours(pd.Series(errors, dtype=float)) == expected


def test_level_and_border_tables(run_dir):
    run = data.load_run(run_dir)
    error = run.area_model - run.area_observed

    levels = data.level_errors(error, run.area_observed)
    positions = data.border_positions(run.flows, "FR", HOURS)

    assert levels.columns.tolist() == [*range(1, 11), "top 5%"]
    assert positions.loc[HOURS[0]].to_dict() == {
        ("DE_LU", "model"): -1.0,
        ("DE_LU", "scheduled"): -1.0,
        ("DE_LU", "ntc out"): 4.0,
        ("DE_LU", "ntc in"): 5.0,
    }


@pytest.mark.parametrize(
    "negative, expected", [(True, "2023-01-09"), (False, "2023-01-02")]
)
def test_hero_week_prefers_widest_week_with_nonpositive_hour(negative, expected):
    prices = pd.Series(50.0, HOURS)
    prices["2023-01-01 12:00"] = 900.0
    prices["2023-01-03 12:00"] = 300.0
    prices["2023-01-10 12:00"] = 150.0
    prices["2023-01-11 03:00"] = -10.0 if negative else 40.0

    week = data.hero_week(prices)

    assert week[0] == pd.Timestamp(expected) and len(week) == data.WEEK_HOURS


def test_puzzle_tables_order_areas_worst_first():
    index = pd.date_range("2023-01-01", periods=20, freq="h")
    observed = pd.DataFrame(
        {"A": [100.0] + [10.0] * 19, "B": [-5.0, 200.0] + [10.0] * 18}, index
    )
    model = pd.DataFrame(
        {"A": [40.0] + [10.0] * 19, "B": [5.0, 150.0] + [12.0] * 18}, index
    )

    tables = data.puzzle_tables(model, observed)

    assert tables["spikes"].model.to_dict() == {"A": -60.0, "B": -50.0}
    assert tables["negative"].to_dict("index") == {
        "B": {"market": 0.05, "model": 0.0},
        "A": {"market": 0.0, "model": 0.0},
    }
    assert tables["cheap"].index.tolist() == ["A", "B"]


def test_net_export(run_dir):
    run = data.load_run(run_dir)

    assert data.net_export(run.flows, "FR", HOURS).to_dict() == pytest.approx(
        {"model": -len(HOURS) / 1e6, "scheduled": -len(HOURS) / 1e6}
    )


def test_total_carrier_errors_sum_areas_with_both_sources():
    hours = HOURS[:2]
    generation = pd.DataFrame(
        {
            ("model", "DE_LU", "gas"): 1e6,
            ("model", "DE_LU", "solar"): 2e6,
            ("model", "FR", "nuclear"): 3e6,
            ("model", "GB", "gas"): 5e6,
            ("entsoe", "DE_LU", "gas"): 1.5e6,
            ("entsoe", "FR", "nuclear"): 3e6,
        },
        hours,
    ).rename_axis(columns=["source", "area", "carrier"])

    table = data.total_carrier_errors(generation, hours)

    assert data.generation_areas(generation).tolist() == ["DE_LU", "FR"]
    expected = pd.DataFrame(
        {
            "model": [2.0, 6.0, 4.0],
            "entsoe": [3.0, 6.0, 0.0],
            "error": [-1.0, 0.0, 4.0],
            "error %": [-100 / 3, 0.0, np.nan],
        },
        ["gas", "nuclear", "solar"],
    )
    pd.testing.assert_frame_equal(table, expected, check_names=False)


def test_story_spike_card_uses_how_it_works_median(run_dir):
    pytest.importorskip("streamlit")
    from dashboard.views.overview import plain_error, puzzles

    run = data.load_run(run_dir)

    spike = plain_error(data.summary(run)[data.TOP_ERROR])
    assert f"is {spike} (median over model regions)" in puzzles(run)[0].answer


@pytest.mark.parametrize(
    "step, result",
    [
        (Step("a", "start", True, 28.5, 9.94), "28.5 / 9.9"),
        (Step("b", "kept", True, 24.0), "24.0 / -"),
        (Step("c", "rejected", False, note="no gain"), "no gain"),
    ],
)
def test_history_result(step, result):
    assert data.history_result(step) == result


def test_history_frame_numbers_steps_and_marks_rejected():
    frame = data.history_frame(HISTORY)

    assert frame.index.tolist() == list(range(1, len(HISTORY) + 1))
    assert frame.kept.iloc[0] and not frame.kept.iloc[-1]
    assert frame.verdict[~frame.kept].str.startswith("rejected").all()
    assert frame["2023"].notna().sum() == sum(s.mae_2023 is not None for s in HISTORY)


def test_zone_names_cover_zone_order():
    assert set(ZONE_ORDER) <= set(NAMES)


def app_test(cache: Path, monkeypatch, **params: str):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(data, "CACHE", cache)
    streamlit.cache_data.clear()
    app = AppTest.from_file(
        str(Path(data.__file__).parent / "app.py"), default_timeout=60
    )
    app.query_params.update(params)
    return app.run()


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_app_renders_pages_and_keeps_state(run_dir, monkeypatch, page):
    app = app_test(run_dir.parent, monkeypatch, page=page, area="FR", period="March")

    assert not app.exception
    assert app.session_state["area"] == "FR" and app.session_state["period"] == "March"
    assert app.query_params["area"] == "FR"


@pytest.mark.slow
def test_story_scores_validation_run_over_same_period(validation_cache, monkeypatch):
    app = app_test(
        validation_cache, monkeypatch, page=STORY_PAGE, run_a="test-run", period="March"
    )

    text = " ".join(m.value for m in app.markdown)
    assert not app.exception
    assert "Validation, March 2019" in text
    assert "validation test-run-2019, March 2019" in text


@pytest.mark.slow
def test_page_buttons_switch_page_and_hidden_state_survives(run_dir, monkeypatch):
    app = app_test(run_dir.parent, monkeypatch, page=ZONE_PAGE, area="FR", agg="daily")
    shown = [s.label for s in app.selectbox]
    app.button(key=f"page_{METHOD_PAGE}").click().run()
    hidden = [s.label for s in app.selectbox]
    app.button(key=f"page_{ZONE_PAGE}").click().run()

    assert not app.exception
    assert shown == ["Run", "Bidding zone", "Period"] and hidden == ["Run"]
    assert app.session_state["page"] == ZONE_PAGE
    assert app.session_state["area"] == "FR" and app.session_state["agg"] == "daily"


@pytest.mark.slow
def test_story_falls_back_and_how_it_works_fills_results(run_dir, monkeypatch):
    story = app_test(run_dir.parent, monkeypatch, page="Overview")
    method = app_test(run_dir.parent, monkeypatch, page=METHOD_PAGE)

    text = " ".join(m.value for m in [*story.markdown, *method.markdown])
    assert not story.exception and not method.exception
    assert story.session_state["page"] == STORY_PAGE
    assert method.table[0].value.columns.tolist() == ["test-run"]
    assert [h.value for h in method.header][-1] == "How the model got better"
    assert "{" not in text and "test-run" in text
