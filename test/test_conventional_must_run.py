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
    n.add("Bus", AREA_TO_BUS.index, country=AREA_TO_BUS.values)
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


@pytest.mark.parametrize(
    "observed, expected",
    [
        ([45.0] * 4, [0.45] * 4),
        ([80.0, 60.0, 95.0, 95.0], [0.6, 0.6, 0.9, 0.9]),
    ],
)
def test_nuclear_min_output_survives_export(
    network, generation, tmp_path, observed, expected
):
    generation[("BE", "nuclear")] = observed
    apply_nuclear_min_output(network, generation, AREA_TO_BUS, {}, {})
    clip_p_min_pu(network)
    network.export_to_netcdf(tmp_path / "network.nc")

    n = pypsa.Network(tmp_path / "network.nc")
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu["BE nuclear"], expected)
    assert (p_min_pu <= n.get_switchable_as_dense("Generator", "p_max_pu")).all().all()


def test_observed_generation_must_cover_snapshots(network, generation):
    with pytest.raises(AssertionError, match="misses 1 snapshots"):
        apply_nuclear_min_output(network, generation.iloc[1:], AREA_TO_BUS, {}, {})


NO_REPORT = pd.Series(dtype=float)
CEILING = [0.5, 0.5, 0.75, 0.75]


def budget(n: pypsa.Network, generation: pd.DataFrame, **kwargs) -> None:
    kwargs = {"monthly_ceiling": True, "reported": NO_REPORT, "fixed": False} | kwargs
    apply_energy_budget(n, generation, AREA_TO_BUS, "biomass", ["BE"], **kwargs)


@pytest.mark.parametrize(
    "p_min_pu, gap, reported, ceiling, e_sum_max, p_max_pu",
    [
        (0.0, False, NO_REPORT, True, [45.0, 15.0], CEILING),
        (0.8, False, NO_REPORT, True, [75.0, 25.0], CEILING),
        (0.0, True, NO_REPORT, True, [52.5, 17.5], CEILING),
        (0.0, False, pd.Series({"PT": 100.0}), True, [75.0, 25.0], CEILING),
        (0.0, False, pd.Series({"PT": 40.0, "BE": 1e3}), True, [45.0, 15.0], CEILING),
        (0.0, False, NO_REPORT, False, [45.0, 15.0], [1.0] * 4),
    ],
)
def test_energy_budget(
    network, generation, p_min_pu, gap, reported, ceiling, e_sum_max, p_max_pu
):
    network.generators.loc[["PT biomass 1", "PT biomass 2"], "p_min_pu"] = p_min_pu
    if gap:
        generation.iloc[0, 2] = np.nan
    budget(network, generation, monthly_ceiling=ceiling, reported=reported)

    g = network.generators
    dense = network.get_switchable_as_dense("Generator", "p_max_pu")
    np.testing.assert_allclose(g.e_sum_max[["PT biomass 1", "PT biomass 2"]], e_sum_max)
    np.testing.assert_allclose(dense["PT biomass 1"], p_max_pu)
    assert g.at["BE biomass", "e_sum_max"] == np.inf
    assert (dense["BE biomass"] == 1.0).all()


FIXED = [0.25, 0.5, 0.75, 0.0]


@pytest.mark.parametrize(
    "gap, reported, p_min_pu, p_max_pu",
    [
        (False, NO_REPORT, FIXED, FIXED),
        (False, pd.Series({"PT": 40.0, "BE": 1e3}), FIXED, FIXED),
        (False, pd.Series({"PT": 72.0, "BE": 1e3}), [0.3, 0.6, 0.9, 0.0], None),
        (False, pd.Series({"PT": 120.0, "BE": 1e3}), [0.5, 1.0, 1.0, 0.0], None),
        (True, NO_REPORT, [0.1, 0.5, 0.75, 0.0], [1.0, 0.5, 0.75, 0.0]),
        (True, pd.Series({"PT": 140.0}), [0.1, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]),
    ],
)
def test_energy_budget_fixed(network, generation, gap, reported, p_min_pu, p_max_pu):
    network.generators.loc[["PT biomass 1", "PT biomass 2"], "p_min_pu"] = 0.1
    if gap:
        generation.iloc[0, 2] = np.nan
    budget(network, generation, monthly_ceiling=False, reported=reported, fixed=True)

    p_min = network.get_switchable_as_dense("Generator", "p_min_pu")
    p_max = network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in ["PT biomass 1", "PT biomass 2"]:
        np.testing.assert_allclose(p_min[gen], p_min_pu)
        np.testing.assert_allclose(p_max[gen], p_max_pu or p_min_pu)
    assert (network.generators.e_sum_max == np.inf).all()
    assert (p_max["BE biomass"] == 1.0).all()
    assert (p_min["BE biomass"] == 0.0).all()


@pytest.mark.parametrize(
    "reported, warning",
    [(pd.Series({"PT": 120.0}), "PT 1 0.000"), (pd.Series({"PT": 2e6}), "PT 3 2.000")],
)
def test_energy_budget_fixed_warns_capped_hours(
    network, generation, caplog, reported, warning
):
    budget(network, generation, monthly_ceiling=False, reported=reported, fixed=True)

    assert f"capped at the capacity, hours and TWh lost: {warning}" in caplog.text


def test_energy_budget_shares_reported_country_over_areas():
    area_to_bus = pd.Series(
        {"DK1": "DK_1", "DK2": "DK_2", "DE0": "DE_LU", "LU0": "DE_LU"}
    )
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", area_to_bus.index, country=["DK", "DK", "DE", "LU"])
    n.add(
        "Generator",
        area_to_bus.index + " biomass",
        bus=area_to_bus.index,
        carrier="biomass",
        p_nom=[30.0, 10.0, 20.0, 20.0],
    )
    columns = pd.MultiIndex.from_product(
        [["DK_1", "DK_2", "DE_LU"], ["biomass"]], names=["area", "carrier"]
    )
    generation = pd.DataFrame(10.0, index=SNAPSHOTS, columns=columns)
    reported = pd.Series({"DK": 200.0, "DE": 40.0, "LU": 40.0})
    apply_energy_budget(
        n, generation, area_to_bus, "biomass", [], False, reported, False
    )

    np.testing.assert_allclose(n.generators.e_sum_max, [150.0, 50.0, 40.0, 40.0])


def test_energy_budget_needs_reported_country(network, generation):
    with pytest.raises(
        AssertionError, match=r"No reported energy for the countries \['PT'\]"
    ):
        budget(network, generation, reported=pd.Series({"BE": 1e3}))


def test_energy_budget_needs_data_in_every_month(network, generation):
    generation.iloc[2:, 2] = np.nan
    with pytest.raises(AssertionError, match=r"misses whole months in \['PT'\]"):
        budget(network, generation)


def test_energy_budget_warns_when_not_binding(network, generation, caplog):
    generation.iloc[:, 2] = [10.0, 20.0, 30.0, 30.0]
    network.generators.loc["PT biomass 1", "p_max_pu"] = 0.1
    budget(network, generation)

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


@pytest.mark.parametrize(
    "dispatch, match",
    [
        (
            {"biomass": {"availability": "monthly", "energy": "budget"}},
            "energy requires availability other than monthly",
        ),
        (
            {"nuclear": {"availability": "monthly", "energy": "budget"}},
            "energy requires availability other than monthly",
        ),
        ({"waste": {"energy": "fixed"}}, r"\['biomass'\] only, not \['waste'\]"),
        ({"coal": {"energy": "eurostat_budget"}}, r"only, not \['coal'\]"),
        ({"coal": {"availability": "hourly"}}, "availability"),
        ({"gas": {"availability": "monthly"}}, "gas"),
        ({"nuclear": {"phase_out": {"DE_LU": "2023-13-01"}}}, "phase_out"),
    ],
)
def test_dispatch_config_invalid(dispatch, match):
    with pytest.raises(ValidationError, match=match):
        BackcastConfig(dispatch=dispatch)


@pytest.mark.parametrize(
    "dispatch",
    [
        {"biomass": {"availability": "outages", "energy": "fixed"}},
        {"coal": {"availability": "monthly"}, "biomass": {"energy": "budget"}},
        {"nuclear": {"availability": "outages", "min_output": "monthly_min"}},
    ],
)
def test_dispatch_config_valid(dispatch):
    BackcastConfig(dispatch=dispatch)
