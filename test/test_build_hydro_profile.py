# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the ENTSO-E water balance in scripts/build_hydro_profile.py."""

import numpy as np
import pandas as pd
import pydantic
import pytest
import xarray as xr

from scripts.build_hydro_profile import (
    balance_inflow,
    entsoe_balance,
    read_filling,
    water_balance,
    zone_filling,
)
from scripts.lib.validation.config.renewable import _HydroConfig

TIME = pd.date_range("2023-01-02", periods=336, freq="h")


@pytest.mark.parametrize(
    "levels, reservoir_mw, ror_mw",
    [
        ((0, 840, 0), (15.0, 5.0), (5.0, 5.0)),
        ((0, 2100, 0), (22.5, 0.0), (5.0, 2.5)),
        ((0, 3000, 0), (4680 / 168, 0.0), (5.0, 0.0)),
        ((0, 0, 336), (9.0, 11.0), (5.0, 5.0)),
    ],
)
def test_water_balance(levels, reservoir_mw, ror_mw):
    hydro = pd.DataFrame({"A": 10.0}, index=TIME)
    ror = pd.DataFrame({"A": 5.0}, index=TIME)
    edges = pd.DatetimeIndex([TIME[0], TIME[168], TIME[-1] + pd.Timedelta("1h")])
    filling = pd.DataFrame({"A": levels}, index=edges, dtype=float)
    ror_t, reservoir_t = water_balance(hydro, ror, filling)
    weekly = (
        pd.DataFrame({"reservoir": reservoir_t.A, "ror": ror_t.A})
        .groupby(TIME >= TIME[168])
        .mean()
    )
    np.testing.assert_allclose(weekly.reservoir, reservoir_mw)
    np.testing.assert_allclose(weekly.ror, ror_mw)


def test_zone_filling():
    filling = pd.DataFrame(
        {"IT": [100.0, 200.0], "FI": [10.0, 20.0], "NO_1": [1.0, 2.0]}
    )
    energy = pd.Series(
        {
            "IT_CNOR": 30.0,
            "IT_NORD": 10.0,
            "IT_SUD": 0.0,
            "FI": 5.0,
            "NO_1": 8.0,
            "NO_2": 8.0,
        }
    )
    expected = pd.DataFrame(
        {
            "IT_CNOR": [75.0, 150.0],
            "IT_NORD": [25.0, 50.0],
            "FI": [10.0, 20.0],
            "NO_1": [1.0, 2.0],
        }
    )
    pd.testing.assert_frame_equal(
        zone_filling(filling, energy).sort_index(axis=1), expected.sort_index(axis=1)
    )


def test_balance_inflow():
    time = pd.date_range("2023-01-01", periods=3, freq="h")
    areas = ["X", "Y", "Z"]
    balance = xr.Dataset(
        {
            "ror": (("time", "area"), np.tile([15.0, 2.0, 1.0], (3, 1))),
            "reservoir": (("time", "area"), np.tile([8.0, 1.0, 2.0], (3, 1))),
            "e_max": ("area", [400.0, 50.0, np.nan]),
            "bus_area": ("bus", areas),
        },
        coords={"time": time, "area": areas, "bus": ["b1", "b2", "b3"]},
    )
    plants = pd.DataFrame(
        {
            "bus": ["b1", "b1", "b1", "b2", "b3", "b4"],
            "carrier": ["ror", "hydro", "hydro", "ror", "hydro", "hydro"],
            "p_nom": [10.0, 20.0, 60.0, 4.0, 5.0, 7.0],
        },
        index=["x_ror", "x_res1", "x_res2", "y_ror", "z_res", "other"],
    )
    inflow, e_nom = balance_inflow(plants, balance)
    expected = pd.Series(
        {"x_ror": 10.0, "y_ror": 3.0, "x_res1": 3.25, "x_res2": 9.75, "z_res": 3.0}
    )
    pd.testing.assert_series_equal(
        inflow.iloc[0].sort_index(), expected.sort_index(), check_names=False
    )
    pd.testing.assert_series_equal(
        e_nom.sort_index(),
        pd.Series({"x_res1": 100.0, "x_res2": 300.0, "z_res": np.nan}),
        check_names=False,
    )


def test_read_filling(tmp_path):
    files = []
    for year, stamps, values in [
        (2022, ["2022-12-19", "2022-12-26"], [1.0, 2.0]),
        (2023, ["2022-12-26", "2023-01-02"], [3.0, 4.0]),
    ]:
        fn = tmp_path / f"filling_{year}.csv"
        pd.Series(values, pd.DatetimeIndex(stamps, name="utc_timestamp")).rename(
            "filling_MWh"
        ).to_csv(fn)
        files.append(str(fn))
    expected = pd.Series(
        [1.0, 3.0, 4.0], pd.DatetimeIndex(["2022-12-26", "2023-01-02", "2023-01-09"])
    )
    pd.testing.assert_series_equal(
        read_filling(files), expected, check_names=False, check_freq=False
    )


AREA_TO_BUS = pd.Series(
    {"b1": "NO_1", "b2": "CH", "b3": "SE_1", "b4": "SE_2", "b5": "DE_LU"},
    name="entsoe_area",
)


def balance_inputs(filling_area: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(
        [
            (a, c)
            for a in ["NO_1", "CH", "SE_1", "SE_2", "DE_LU"]
            for c in ["hydro", "ror"]
        ]
    )
    generation = pd.DataFrame(
        [[10.0, 5.0, 3.0, 1.0, np.nan, np.nan, 4.0, 0.0, 2.0, 2.0]],
        index=TIME,
        columns=columns,
    )
    filling = pd.DataFrame(
        {filling_area: [800.0, 1000.0]},
        index=pd.DatetimeIndex([TIME[0], TIME[-1] + pd.Timedelta("1h")]),
    )
    return generation, filling


def test_entsoe_balance():
    generation, filling = balance_inputs("NO_1")
    ds = entsoe_balance(
        generation, filling, AREA_TO_BUS, ["NO", "CH", "SE"], ["CH"], TIME
    )
    assert list(ds.indexes["area"]) == ["NO_1", "SE_2"]
    np.testing.assert_allclose(ds["e_max"], [1000.0, np.nan])
    np.testing.assert_allclose(ds["reservoir"].mean("time"), [10.0, 4.0])
    np.testing.assert_allclose(ds["ror"].mean("time"), [5.0, 0.0])


@pytest.mark.parametrize(
    "exclude, filling_area", [(["XX"], "NO_1"), (["CH"], "NO_9"), (["CH"], "XX")]
)
def test_entsoe_balance_unknown_areas(exclude, filling_area):
    generation, filling = balance_inputs(filling_area)
    with pytest.raises(AssertionError, match="Unknown ENTSO-E areas"):
        entsoe_balance(generation, filling, AREA_TO_BUS, ["NO"], exclude, TIME)


@pytest.mark.parametrize(
    "source, valid", [("eia", True), ("entsoe_balance", True), ("glaes", False)]
)
def test_inflow_source_validation(source, valid):
    if valid:
        assert _HydroConfig(inflow_source=source).inflow_source == source
    else:
        with pytest.raises(pydantic.ValidationError):
            _HydroConfig(inflow_source=source)
