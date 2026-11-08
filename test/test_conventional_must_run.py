# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the nuclear minimum output, energy budgets and fuel prices in scripts/lib/conventional.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.lib.conventional import (
    apply_energy_budget,
    apply_nuclear_min_output,
    clip_p_min_pu,
    set_fuel_price,
)
from scripts.lib.validation.config.backcast import BackcastConfig

SNAPSHOTS = pd.date_range("2023-01-31 22:00", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"BE0": "BE", "FR0": "FR", "NL0": "NL", "PT0": "PT"})


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index)
    n.add(
        "Generator",
        ["BE nuclear", "FR nuclear", "NL nuclear"],
        bus=["BE0", "FR0", "NL0"],
        carrier="nuclear",
        p_nom=[100.0, 200.0, 50.0],
        p_min_pu=0.3,
        p_max_pu=0.9,
    )
    n.generators_t.p_max_pu["FR nuclear"] = [0.5, 0.5, 1.0, 1.0]
    n.add(
        "Generator",
        ["PT biomass 1", "PT biomass 2", "BE biomass"],
        bus=["PT0", "PT0", "BE0"],
        carrier="biomass",
        p_nom=[30.0, 10.0, 20.0],
    )
    return n


@pytest.fixture
def generation() -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [("BE", "nuclear"), ("FR", "nuclear"), ("PT", "biomass"), ("BE", "biomass")],
        names=["area", "carrier"],
    )
    values = [
        [80.0, 10.0, 10.0, 5.0],
        [60.0, 10.0, 20.0, 5.0],
        [np.nan, 10.0, 30.0, 5.0],
        [np.nan, 10.0, 0.0, 5.0],
    ]
    return pd.DataFrame(values, index=SNAPSHOTS, columns=columns)


@pytest.mark.parametrize(
    "area_map, be, nl",
    [
        ({}, [0.6, 0.6, 0.0, 0.0], [0.3] * 4),
        ({"NL": "BE"}, [0.4, 0.4, 0.0, 0.0], [0.4, 0.4, 0.0, 0.0]),
    ],
)
def test_nuclear_min_output(network, generation, area_map, be, nl):
    apply_nuclear_min_output(network, generation, AREA_TO_BUS, area_map, {"FR": 0.2})

    p_min_pu = network.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu["BE nuclear"], be)
    np.testing.assert_allclose(p_min_pu["NL nuclear"], nl)
    np.testing.assert_allclose(p_min_pu["FR nuclear"], [0.1, 0.1, 0.2, 0.2])


def test_nuclear_min_output_survives_export(network, generation, tmp_path):
    generation[("BE", "nuclear")] = 45.0
    apply_nuclear_min_output(network, generation, AREA_TO_BUS, {}, {})
    clip_p_min_pu(network)
    network.export_to_netcdf(tmp_path / "network.nc")

    n = pypsa.Network(tmp_path / "network.nc")
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu["BE nuclear"], 0.45)
    assert (p_min_pu <= n.get_switchable_as_dense("Generator", "p_max_pu")).all().all()


def test_observed_generation_must_cover_snapshots(network, generation):
    with pytest.raises(AssertionError, match="misses 1 snapshots"):
        apply_nuclear_min_output(network, generation.iloc[1:], AREA_TO_BUS, {}, {})


@pytest.mark.parametrize(
    "p_min_pu, gap, e_sum_max",
    [
        (0.0, False, [45.0, 15.0]),
        (0.8, False, [75.0, 25.0]),
        (0.0, True, [52.5, 17.5]),
    ],
)
def test_energy_budget(network, generation, p_min_pu, gap, e_sum_max):
    network.generators.loc[["PT biomass 1", "PT biomass 2"], "p_min_pu"] = p_min_pu
    if gap:
        generation.iloc[0, 2] = np.nan
    apply_energy_budget(network, generation, AREA_TO_BUS, "biomass", ["BE"])

    g = network.generators
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    np.testing.assert_allclose(g.e_sum_max[["PT biomass 1", "PT biomass 2"]], e_sum_max)
    np.testing.assert_allclose(p_max_pu["PT biomass 1"], [0.5, 0.5, 0.75, 0.75])
    assert g.at["BE biomass", "e_sum_max"] == np.inf
    assert (p_max_pu["BE biomass"] == 1.0).all()


def test_energy_budget_needs_data_in_every_month(network, generation):
    generation.iloc[2:, 2] = np.nan
    with pytest.raises(AssertionError, match=r"misses whole months in \['PT'\]"):
        apply_energy_budget(network, generation, AREA_TO_BUS, "biomass", ["BE"])


def test_energy_budget_warns_when_not_binding(network, generation, caplog):
    generation.iloc[:, 2] = [10.0, 20.0, 30.0, 30.0]
    network.generators.loc["PT biomass 1", "p_max_pu"] = 0.1
    apply_energy_budget(network, generation, AREA_TO_BUS, "biomass", ["BE"])

    assert "exceeds the available energy, not binding, by TWh: PT" in caplog.text


@pytest.mark.parametrize(
    "price, fuel, marginal_cost",
    [({"nuclear": 1.31}, 1.31, 4.459 + 1.31 / 0.326), ({}, 7.4536, 27.32)],
)
def test_set_fuel_price(price, fuel, marginal_cost):
    costs = pd.DataFrame(
        {
            "VOM": [4.459],
            "fuel": [7.4536],
            "efficiency": [0.326],
            "marginal_cost": [27.32],
        },
        index=["nuclear"],
    )
    set_fuel_price(costs, price)

    assert costs.at["nuclear", "fuel"] == pytest.approx(fuel)
    assert costs.at["nuclear", "marginal_cost"] == pytest.approx(marginal_cost)


def test_set_fuel_price_unknown_technology():
    costs = pd.DataFrame(
        {"VOM": [0.0], "fuel": [9.35], "efficiency": [0.468]}, index=["biomass"]
    )
    with pytest.raises(AssertionError, match="wood"):
        set_fuel_price(costs, {"wood": 30.0})


def test_energy_budget_carrier_not_monthly_available():
    with pytest.raises(ValidationError, match="energy_budget_carriers"):
        BackcastConfig(
            energy_budget_carriers=["biomass"],
            monthly_availability_carriers=["biomass"],
        )
