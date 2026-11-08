# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Tests for selected helpers in scripts/add_electricity.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr

from scripts.add_electricity import (
    attach_hydro,
    attach_load,
    attach_storageunits,
    attach_stores,
    cap_phs_max_hours,
    estimate_efficiency,
    read_irenastat,
)
from scripts.lib.validation.config.conventional import _EstimateEfficienciesConfig


def test_attach_load(tmp_path):
    """Clustered demand is attached per bus with scaling applied."""

    times = pd.date_range("2000-01-01", periods=2, freq="h")
    buses = ["zone_1", "zone_2"]
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    data = xr.DataArray(
        values,
        coords={"time": times, "bus": buses},
        dims=["time", "bus"],
        name="electricity demand (MW)",
    )
    load_path = tmp_path / "electricity_demand.nc"
    data.to_netcdf(load_path)

    n = pypsa.Network()
    n.set_snapshots(times)
    n.add("Bus", buses)

    attach_load(n, load_path.as_posix(), scaling=2.0)

    assert sorted(n.loads.index) == buses
    assert sorted(n.loads_t.p_set.columns) == buses
    np.testing.assert_allclose(n.loads_t.p_set[buses].values, 2.0 * values)


def test_attach_storageunits_energy_basis():
    """A dispatched-basis `max_hours` sizes the store to sustain it, at no extra cost."""
    costs = pd.DataFrame(
        {
            "capital_cost": {"iron-air": 157959.0, "battery": 60630.0},
            "marginal_cost": {"iron-air": 0.0, "battery": 0.0},
            "lifetime": {"iron-air": 17.5, "battery": 17.5},
            "efficiency": {
                "iron-air battery charge": 0.74,
                "iron-air battery discharge": 0.63,
                "battery inverter": 0.96,
            },
        }
    )
    max_hours = {"iron-air": 100, "battery": 6}

    n = pypsa.Network()
    n.add("Bus", ["bus_1", "bus_2"])
    attach_storageunits(n, costs, n.buses.index, ["iron-air", "battery"], max_hours)

    su = n.storage_units.set_index("carrier")

    iron_air = su.loc["iron-air"]
    np.testing.assert_allclose(iron_air.max_hours * iron_air.efficiency_dispatch, 100)
    np.testing.assert_allclose(iron_air.capital_cost, 157959.0)

    # Stored-basis carriers must be left alone, despite efficiency_dispatch < 1.
    np.testing.assert_allclose(su.loc["battery"].max_hours, 6)


def test_attach_stores_energy_basis():
    """A dispatched-basis store cost is converted to cost per MWh stored."""
    costs = pd.DataFrame(
        {
            "capital_cost": {
                "iron-air battery": 1000.0,
                "iron-air battery charge": 0.0,
                "iron-air battery discharge": 0.0,
            },
            "marginal_cost": 0.0,
            "lifetime": 17.5,
            "efficiency": {
                "iron-air battery": 1.0,
                "iron-air battery charge": 0.74,
                "iron-air battery discharge": 0.63,
            },
        }
    )

    n = pypsa.Network()
    n.add("Bus", ["bus_1"])
    attach_stores(n, costs, n.buses.index, ["iron-air"])

    np.testing.assert_allclose(n.stores.capital_cost, 630.0)


def test_estimate_efficiency():
    """Efficiency rises with build year (retrofit first), is clipped and degrades with age."""
    ppl = pd.DataFrame(
        {
            "carrier": ["CCGT", "coal", "coal", "biomass", "CCGT"],
            "datein": [2000, 1950, 1970, 2000, np.nan],
            "dateretrofit": [np.nan, np.nan, 2010, np.nan, np.nan],
        }
    )
    config = _EstimateEfficienciesConfig().model_dump()

    eta = estimate_efficiency(ppl, config)

    expected = [
        0.48 * (1 - 0.015),
        0.28 * (1 - 0.065),
        0.43 * (1 - 0.005),
        np.nan,
        np.nan,
    ]
    np.testing.assert_allclose(eta, expected)


@pytest.mark.parametrize(
    ("source", "ror_pu", "res_inflow", "res_max_hours", "res_efficiency"),
    [("eia", 0.75, 22.5, 80e3 / 30, 0.9), ("entsoe_balance", 0.4, 6.0, 20.0, 1.0)],
)
def test_attach_hydro(
    tmp_path, source, ror_pu, res_inflow, res_max_hours, res_efficiency
):
    """Balance reservoirs take energy capacity and inflow from the water balance; others keep the EIA inflow."""
    times = pd.date_range("2023-01-01", periods=3, freq="h")
    inflow = xr.DataArray(
        [[30.0, 20.0]] * 3,
        coords={"time": times, "countries": ["NO", "SE"]},
        name="inflow",
    )
    profile = tmp_path / "profile_hydro.nc"
    if source == "eia":
        inflow.to_netcdf(profile)
    else:
        xr.Dataset(
            {
                "inflow": inflow,
                "ror": (("time", "area"), [[4.0]] * 3),
                "reservoir": (("time", "area"), [[6.0]] * 3),
                "e_max": ("area", [600.0]),
                "bus_area": ("bus", ["NO_1", "SE_1"]),
            },
            coords={"time": times, "area": ["NO_1"], "bus": ["b1", "b2"]},
        ).to_netcdf(profile)
    capacities = tmp_path / "hydro_capacities.csv"
    capacities.write_text(
        "Country,p_nom_discharge[GW],E_store[TWh]\nNO,30,80\nSE,15,30\n"
    )
    costs = pd.DataFrame(
        {
            "efficiency": 0.9,
            "capital_cost": 0.0,
            "marginal_cost": 0.0,
            "CO2 intensity": 0.0,
        },
        index=["ror", "hydro"],
    )
    ppl = pd.DataFrame(
        {
            "bus": ["b1", "b1", "b2"],
            "country": ["NO", "NO", "SE"],
            "carrier": ["ror", "hydro", "hydro"],
            "p_nom": [10.0, 30.0, 15.0],
            "max_hours": np.nan,
        },
        index=["ror_x", "res_x", "res_y"],
    )
    n = pypsa.Network()
    n.set_snapshots(times)
    n.add("Bus", ["b1", "b2"], country=["NO", "SE"])

    attach_hydro(
        n,
        costs,
        ppl,
        str(profile),
        str(capacities),
        [],
        ["ror", "hydro"],
        inflow_source=source,
        hydro_max_hours="estimate_by_large_installations",
    )

    su = n.storage_units
    np.testing.assert_allclose(n.generators_t.p_max_pu["ror_x"], ror_pu)
    np.testing.assert_allclose(n.storage_units_t.inflow["res_x"], res_inflow)
    np.testing.assert_allclose(n.storage_units_t.inflow["res_y"], 20.0)
    np.testing.assert_allclose(
        su.loc[["res_x", "res_y"], "max_hours"], [res_max_hours, 2000.0]
    )
    np.testing.assert_allclose(
        su.loc[["res_x", "res_y"], "efficiency_dispatch"], [res_efficiency, 0.9]
    )


@pytest.mark.parametrize(
    ("country", "max_hours", "expected"),
    [("AT", 50.0, 20.0), ("AT", 8.0, 8.0), ("DK", 50.0, 50.0), ("FR", 50.0, 50.0)],
)
def test_cap_phs_max_hours(tmp_path, country, max_hours, expected):
    """PHS max hours are capped at the country value; countries without data keep theirs."""
    fn = tmp_path / "phs.csv"
    fn.write_text(
        "# source\ncountry,n,p_nom_discharge,p_nom_charge,e_stor\n"
        "AT,1,2.0,1.5,40.0\nDK,0,-,-,-\n"
    )
    idx = pd.Index(["phs"])
    capped = cap_phs_max_hours(
        pd.Series(max_hours, idx), pd.Series(country, idx), str(fn)
    )
    assert capped.at["phs"] == expected


def test_read_irenastat(tmp_path):
    """An IRENASTAT file in the powerplantmatching v0.9 layout reads with the technology names of the bundled file."""
    fn = tmp_path / "irenastat.csv"
    fn.write_text(
        "Country/area,Technology,Grid connection,Year,Electricity capacity statistics\n"
        "Germany,Onshore wind energy,OnGrid,2025,60000\n"
        "Germany,Solar photovoltaic,OnGrid,2025,100000\n"
        "Germany,Solar photovoltaic,OffGrid,2025,\n"
        "France,Offshore wind energy,OnGrid,2025,1800\n"
    )

    df = read_irenastat(str(fn))

    capacities = df.groupby(["Technology", "Country"]).Capacity.sum()
    assert capacities.to_dict() == {
        ("Offshore", "FR"): 1800,
        ("Onshore", "DE"): 60000,
        ("PV", "DE"): 100000,
    }
    assert (df.Year == 2025).all()
