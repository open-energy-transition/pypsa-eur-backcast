# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the heat-led CHP floor in scripts/build_chp_floor.py and scripts/lib/conventional.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.build_chp_floor import (
    floor_energy,
    heat_profile,
    read_chp_production,
)
from scripts.lib.conventional import apply_chp_floor, clip_p_min_pu
from scripts.lib.eurostat import eurostat_year, gross_electricity

SNAPSHOTS = pd.date_range("2023-01-01", periods=72, freq="h")
TSV = """freq,nrg_bal,siec,unit,geo\\TIME_PERIOD\t2022 \t2023
A,GEP_MAPCHP,G3000,GWH,DE\t9 \t10
A,GHP_MAPCHP,G3000,GWH,DE\t9 \t20
A,GEP_MAPCHP,R5300,GWH,EL\t: \t4 p
A,GHP_MAPCHP,R5300,GWH,EL\t: \t8
A,GEP_MAPCHP,N900H,GWH,DE\t0 \t0
A,GEP_MAPE,G3000,GWH,DE\t50 \t50
"""


def test_read_chp_production(tmp_path):
    path = tmp_path / "nrg_bal_peh.tsv"
    path.write_text(TSV)

    production = read_chp_production(str(path), 2023, ["gas", "biomass"])

    expected = pd.DataFrame(
        {"electricity": [10e3, 4e3], "heat": [20e3, 8e3]},
        index=pd.MultiIndex.from_tuples(
            [("DE", "gas"), ("GR", "biomass")], names=["country", "fuel"]
        ),
    )
    pd.testing.assert_frame_equal(
        production.sort_index(), expected.sort_index(), check_names=False
    )


def test_read_chp_production_missing_year(tmp_path):
    path = tmp_path / "nrg_bal_peh.tsv"
    path.write_text(TSV)

    with pytest.raises(AssertionError, match="no data for 2025"):
        read_chp_production(str(path), 2025, ["gas"])


@pytest.mark.parametrize(
    ("siec", "expected"),
    [("O4680", 3e3), ("O4671XR5220B", 3e3), ("O4610", None), ("O4699", None)],
    ids=["fuel-oil", "gas-oil", "refinery-gas", "other-oil-products"],
)
def test_read_chp_production_oil(tmp_path, siec, expected):
    path = tmp_path / "nrg_bal_peh.tsv"
    path.write_text(
        "freq,nrg_bal,siec,unit,geo\\TIME_PERIOD\t2023\n"
        f"A,GEP_MAPCHP,{siec},GWH,IT\t3\n"
        f"A,GHP_MAPCHP,{siec},GWH,IT\t5\n"
        "A,GEP_MAPCHP,O4000XBIO,GWH,IT\t3\n"
        "A,GEP_MAPCHP,G3000,GWH,IT\t1\n"
    )

    production = read_chp_production(str(path), 2023, ["gas", "oil"])

    assert production["electricity"].get(("IT", "oil")) == expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([("GEP", "BIOE", "DE", "50"), ("GEP", "W6210", "DE", "6")], {"DE": 44e3}),
        ([("GEP", "BIOE", "EL", "2 p"), ("GEP", "W6210", "FI", "1")], {"GR": 2e3}),
        ([("GEP", "BIOE", "FI", ":"), ("GEP", "W6210", "FI", "1")], {}),
        ([("GEP_MAPCHP", "BIOE", "DE", "50"), ("GEP", "R5300", "DE", "5")], {}),
    ],
    ids=["net-of-renewable-waste", "country-codes", "no-biomass", "other-rows"],
)
def test_gross_electricity(tmp_path, rows, expected):
    path = tmp_path / "nrg_bal_peh.tsv"
    lines = [f"A,{bal},{siec},GWH,{geo}\t{value}\n" for bal, siec, geo, value in rows]
    path.write_text("freq,nrg_bal,siec,unit,geo\\TIME_PERIOD\t2023\n" + "".join(lines))

    reported = gross_electricity(str(path), 2023, "biomass")

    pd.testing.assert_series_equal(
        reported,
        pd.Series(expected, dtype=float),
        check_names=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(("year", "expected"), [(None, 2023), (2022, 2022)])
def test_eurostat_year(year, expected):
    assert eurostat_year(SNAPSHOTS, year) == expected


def test_eurostat_year_multiple_years():
    snapshots = pd.date_range("2022-12-31", periods=48, freq="h")

    with pytest.raises(AssertionError, match="one year"):
        eurostat_year(snapshots, 2022)


@pytest.mark.parametrize(
    ("electricity", "heat", "c_b", "expected"),
    [(10.0, 20.0, 0.35, 7.0), (10.0, 20.0, 0.96, 10.0)],
    ids=["heat-limited", "electricity-limited"],
)
def test_floor_energy(electricity, heat, c_b, expected):
    production = pd.DataFrame(
        {"electricity": [electricity], "heat": [heat]},
        index=pd.MultiIndex.from_tuples([("DE", "gas")], names=["country", "fuel"]),
    )

    energy = floor_energy(production, pd.Series({"gas": c_b}))

    assert energy[("DE", "gas")] == pytest.approx(expected)


@pytest.mark.parametrize("hot_water_share", [0.0, 0.2, 0.5])
def test_heat_profile(hot_water_share):
    temperature = pd.DataFrame(
        {"DE": np.repeat([5.0, 10.0, 20.0], 24)}, index=SNAPSHOTS
    )

    profile = heat_profile(temperature, SNAPSHOTS, 15.0, hot_water_share)

    daily = profile.DE.resample("D").sum()
    flat = hot_water_share / 3
    np.testing.assert_allclose(
        daily, [flat + (1 - hot_water_share) * w for w in [2 / 3, 1 / 3, 0]]
    )
    assert profile.DE.groupby(SNAPSHOTS.floor("D")).nunique().eq(1).all()


@pytest.mark.parametrize(
    ("replace", "expected_biomass"), [(["biomass"], 0.1), ([], 0.3)]
)
def test_apply_chp_floor(replace, expected_biomass):
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS[:2])
    n.add("Bus", ["DE00", "GB00"], country=["DE", "GB"])
    n.add(
        "Generator",
        ["DE CCGT", "DE OCGT", "DE biomass", "GB biomass"],
        bus=["DE00", "DE00", "DE00", "GB00"],
        carrier=["CCGT", "OCGT", "biomass", "biomass"],
        p_nom=[300.0, 100.0, 100.0, 100.0],
        p_min_pu=[0.0, 0.0, 0.3, 0.3],
    )
    floor = pd.DataFrame(
        {("DE", "gas"): [200.0, 40.0], ("DE", "biomass"): [10.0, 10.0]},
        index=SNAPSHOTS[:2],
    )

    apply_chp_floor(n, floor, replace)

    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu["DE CCGT"], [0.5, 0.1])
    np.testing.assert_allclose(p_min_pu["DE OCGT"], [0.5, 0.1])
    np.testing.assert_allclose(p_min_pu["DE biomass"], expected_biomass)
    np.testing.assert_allclose(p_min_pu["GB biomass"], 0.3)


@pytest.fixture
def chp_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS[:2])
    n.add("Bus", "IT00", country="IT")
    n.add(
        "Generator",
        ["IT CCGT", "IT oil"],
        bus="IT00",
        carrier=["CCGT", "oil"],
        p_nom=[100.0, 100.0],
        p_max_pu=pd.DataFrame(
            {"IT CCGT": [1.0, 1.0], "IT oil": [0.8, 0.5]}, index=SNAPSHOTS[:2]
        ),
    )
    return n


@pytest.mark.parametrize(
    ("oil_floor", "expected"),
    [([50.0, 40.0], [0.5, 0.4]), ([120.0, 60.0], [0.8, 0.5])],
    ids=["below-p_max_pu", "above-p_max_pu"],
)
def test_apply_chp_floor_then_clip(chp_network, oil_floor, expected):
    floor = pd.DataFrame({("IT", "oil"): oil_floor}, index=SNAPSHOTS[:2])

    apply_chp_floor(chp_network, floor, [])
    clip_p_min_pu(chp_network)

    p_min_pu = chp_network.get_switchable_as_dense("Generator", "p_min_pu")
    p_max_pu = chp_network.get_switchable_as_dense("Generator", "p_max_pu")
    np.testing.assert_allclose(p_min_pu["IT oil"], expected)
    np.testing.assert_allclose(p_min_pu["IT CCGT"], 0.0)
    assert (p_min_pu <= p_max_pu).all().all()


@pytest.mark.parametrize(
    ("coal_floor", "warned"), [(0.0, False), (10.0, True)], ids=["empty", "positive"]
)
def test_apply_chp_floor_unmatched(chp_network, caplog, coal_floor, warned):
    floor = pd.DataFrame(
        {("IT", "gas"): [10.0, 10.0], ("IT", "coal"): coal_floor}, index=SNAPSHOTS[:2]
    )

    apply_chp_floor(chp_network, floor, [])

    assert ("CHP floor without generators" in caplog.text) == warned
    np.testing.assert_allclose(
        chp_network.get_switchable_as_dense("Generator", "p_min_pu")["IT CCGT"], 0.1
    )
