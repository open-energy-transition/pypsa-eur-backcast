# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the heat-led CHP floor in scripts/build_chp_floor.py and scripts/lib/conventional.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.build_chp_floor import (
    eurostat_year,
    floor_energy,
    heat_profile,
    read_chp_production,
)
from scripts.lib.conventional import apply_chp_floor

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
