# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the ENTSO-E unavailability retrieval and scripts/lib/unavailability.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.lib.unavailability import (
    apply_unavailability,
    hourly_mean,
    hourly_unavailable,
    latest_outages,
    match_production_units,
    name_score,
    production_units,
    unavailable_share,
)
from scripts.retrieve_unavailability_entsoe import outage_steps

HOURS = pd.date_range("2023-01-01", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"DE00": "DE_LU", "FI00": "FI"})


def step(**kwargs) -> dict:
    defaults = dict(
        doctype="A80",
        mrid="m1",
        revision=1,
        docstatus=None,
        production_resource_id="P1",
        production_resource_name="Plant",
        unit_name="Plant B1",
        plant_type="Fossil Gas",
        nominal_power=100.0,
        avail_qty=0.0,
        start=pd.Timestamp("2023-01-01 00:00", tz="UTC"),
        end=pd.Timestamp("2023-01-01 02:00", tz="UTC"),
        area="DE_LU",
    )
    return {**defaults, **kwargs}


def steps(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def utc(time: str) -> pd.Timestamp:
    return pd.Timestamp(f"2023-01-01 {time}", tz="UTC")


@pytest.fixture
def ppl() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Isar 2 Kki 2", "Gas A", "Gas B", "Gas C", "Olkiluoto"],
            "carrier": ["nuclear", "CCGT", "CCGT", "CCGT", "nuclear"],
            "p_nom": [1410.0, 100.0, 300.0, 200.0, 1000.0],
            "bus": ["DE00", "DE00", "DE00", "DE00", "FI00"],
            "area": ["DE_LU", "DE_LU", "DE_LU", "DE_LU", "FI"],
            "projectid": [
                np.nan,
                "{'ENTSOE': {'P1'}}",
                "{'ENTSOE': {'P1'}, 'GEM': {'G1'}}",
                "{'GEM': {'G2'}}",
                "{'ENTSOE': {'P9'}}",
            ],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_outage_steps_renames_and_converts_to_utc():
    raw = pd.DataFrame(
        {
            "mrid": ["m1"],
            "revision": [2],
            "docstatus": [None],
            "businesstype": ["Unplanned outage"],
            "production_resource_id": ["P1"],
            "production_resource_name": ["Plant"],
            "production_resource_psr_name": ["Plant B1"],
            "plant_type": ["Nuclear"],
            "nominal_power": [890.0],
            "avail_qty": ["585"],
            "start": [pd.Timestamp("2023-03-10 21:15", tz="Europe/Helsinki")],
            "end": [pd.Timestamp("2023-03-10 22:45", tz="Europe/Helsinki")],
        },
        index=pd.Index([pd.Timestamp("2023-03-08", tz="Europe/Helsinki")]),
    )
    result = outage_steps(raw, "A80")
    assert result.loc[0, "unit_name"] == "Plant B1"
    assert result.loc[0, "avail_qty"] == 585
    assert result.loc[0, "start"] == pd.Timestamp("2023-03-10 19:15", tz="UTC")
    assert result.loc[0, "doctype"] == "A80"


def test_latest_outages_keeps_last_revision_and_drops_cancelled():
    result = latest_outages(
        steps(
            step(mrid="m1", revision=1, avail_qty=0.0),
            step(mrid="m1", revision=2, avail_qty=50.0),
            step(mrid="m2", revision=1),
            step(mrid="m2", revision=2, docstatus="Cancelled"),
        )
    )
    assert result.mrid.tolist() == ["m1"]
    assert result.avail_qty.tolist() == [50.0]


def test_latest_outages_prefers_generation_units():
    result = latest_outages(
        steps(
            step(doctype="A80", production_resource_id="P1"),
            step(doctype="A77", mrid="m2", production_resource_id="P1"),
            step(doctype="A77", mrid="m3", production_resource_id="P2"),
        )
    )
    assert sorted(zip(result.doctype, result.production_resource_id)) == [
        ("A77", "P2"),
        ("A80", "P1"),
    ]


def test_hourly_mean_weights_partial_hours():
    level = pd.Series(
        [100.0, 0.0],
        index=pd.DatetimeIndex(["2022-12-31 23:00", "2023-01-01 01:30"]),
    )
    result = hourly_mean(level, HOURS)
    np.testing.assert_allclose(result.values, [100.0, 50.0, 0.0, 0.0])


def test_hourly_mean_skips_dropped_hours():
    hours = HOURS.delete(1)
    level = pd.Series([100.0], index=pd.DatetimeIndex(["2023-01-01 00:00"]))
    result = hourly_mean(level, hours)
    np.testing.assert_allclose(result.values, [100.0, 100.0, 100.0])


def test_hourly_unavailable_takes_maximum_of_overlapping_outages_per_unit():
    result = hourly_unavailable(
        steps(
            step(mrid="m1", avail_qty=60.0, end=utc("03:00")),
            step(mrid="m2", avail_qty=0.0, start=utc("01:00"), end=utc("02:00")),
            step(mrid="m3", unit_name="Plant B2", start=utc("02:30"), end=utc("05:00")),
        ),
        HOURS,
    )
    np.testing.assert_allclose(result["P1"].values, [40.0, 100.0, 90.0, 100.0])


def test_hourly_unavailable_uses_production_unit_without_unit_name():
    result = hourly_unavailable(
        steps(step(doctype="A77", unit_name=np.nan, avail_qty=25.0)), HOURS
    )
    np.testing.assert_allclose(result["P1"].values, [75.0, 75.0, 0.0, 0.0])


def test_production_units_sum_generation_units():
    result = production_units(
        steps(
            step(mrid="m1", unit_name="Plant B1", nominal_power=100.0),
            step(mrid="m2", unit_name="Plant B1", nominal_power=100.0),
            step(mrid="m3", unit_name="Plant B2", nominal_power=50.0),
        )
    )
    assert result.loc["P1", "capacity"] == 150.0
    assert result.loc["P1", "names"] == ["Plant", "Plant B1", "Plant B2"]


def units(**rows: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient="index")


def test_match_production_units_by_code_and_name(ppl):
    result = match_production_units(
        units(
            P1=dict(
                area="DE_LU", plant_type="Fossil Gas", capacity=400.0, names=["Gas"]
            ),
            X1=dict(
                area="DE_LU", plant_type="Nuclear", capacity=1400.0, names=["Isar 2"]
            ),
        ),
        ppl,
    )
    assert sorted(result.items()) == [("P1", 11), ("P1", 12), ("X1", 10)]


@pytest.mark.parametrize(
    "unit",
    [
        dict(area="DE_LU", plant_type="Nuclear", capacity=900.0, names=["Isar 2"]),
        dict(area="FI", plant_type="Nuclear", capacity=1400.0, names=["Isar 2"]),
        dict(area="DE_LU", plant_type="Fossil Gas", capacity=1400.0, names=["Isar 2"]),
        dict(area="DE_LU", plant_type="Nuclear", capacity=1400.0, names=["Emsland"]),
    ],
)
def test_match_production_units_rejects_capacity_area_fuel_and_name(ppl, unit):
    assert match_production_units(units(X1=unit), ppl).empty


@pytest.mark.parametrize(
    "names, candidate, score",
    [
        (
            ["Kernkraftwerk Neckarwestheim", "KKW Neckarwestheim 2"],
            "Neckarwestheim 2 Gkn 2",
            1.0,
        ),
        (["HKW Mitte"], "Berlin Mitte", 1.0),
        (["HKW Merkenich"], "Gud", 0.0),
        (["GuD Ludwigshafen Mitte", "GT 2"], "Gt 2", 0.0),
    ],
)
def test_name_score_ignores_generic_words_and_unit_numbers(names, candidate, score):
    assert name_score(names, candidate) == score


def test_unavailable_share_splits_clips_and_covers(ppl):
    unavailable = pd.DataFrame({"P1": [0.0, 200.0, 600.0, 0.0]}, index=HOURS)
    matches = pd.Series([11, 12], index=["P1", "P1"])
    share, coverage = unavailable_share(unavailable, matches, ppl, ["CCGT", "nuclear"])
    np.testing.assert_allclose(
        share[("DE00", "CCGT")].values, [0, 200 / 600, 400 / 600, 0]
    )
    assert (share[("FI00", "nuclear")] == 0).all()
    assert coverage.loc[("DE_LU", "CCGT"), "coverage"] == pytest.approx(400 / 600)
    assert coverage.loc[("DE_LU", "nuclear"), "coverage"] == 0


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(HOURS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    n.add(
        "Generator",
        ["DE00 CCGT Q0 efficiency", "DE00 CCGT", "DE00 nuclear", "FI00 nuclear"],
        bus=["DE00", "DE00", "DE00", "FI00"],
        carrier=["CCGT", "CCGT", "nuclear", "nuclear"],
        p_nom=100.0,
    )
    n.generators_t.p_max_pu["FI00 nuclear"] = 0.7
    return n


def test_apply_unavailability(network):
    columns = pd.MultiIndex.from_tuples(
        [("DE00", "CCGT"), ("DE00", "nuclear"), ("FI00", "nuclear")]
    )
    share = pd.DataFrame(0.2, index=HOURS, columns=columns)
    coverage = pd.DataFrame(
        {"coverage": [0.5, 1.0, 0.5]},
        index=pd.MultiIndex.from_tuples(
            [("DE_LU", "CCGT"), ("DE_LU", "nuclear"), ("FI", "nuclear")]
        ),
    )
    apply_unavailability(
        network,
        share,
        coverage,
        AREA_TO_BUS,
        {"nuclear": 0.9},
        {"DE_LU": "2022-12-31"},
    )
    p_max_pu = network.generators_t.p_max_pu
    assert (p_max_pu[["DE00 CCGT Q0 efficiency", "DE00 CCGT"]] == 0.8).all().all()
    assert (p_max_pu["DE00 nuclear"] == 0.0).all()
    assert (p_max_pu["FI00 nuclear"] == 0.7).all()


def test_apply_unavailability_requires_all_snapshots(network):
    share = pd.DataFrame(
        0.2, index=HOURS[:2], columns=pd.MultiIndex.from_tuples([("DE00", "CCGT")])
    )
    coverage = pd.DataFrame(
        {"coverage": [1.0]}, index=pd.MultiIndex.from_tuples([("DE_LU", "CCGT")])
    )
    with pytest.raises(AssertionError, match="all snapshots"):
        apply_unavailability(network, share, coverage, AREA_TO_BUS, {}, {})
