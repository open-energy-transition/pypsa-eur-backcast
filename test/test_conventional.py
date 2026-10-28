# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the generator limits and costs in scripts/lib/conventional.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.add_electricity import attach_conventional_generators
from scripts.lib.conventional import (
    cap_fuel_price,
    clip_p_min_pu,
    spread_marginal_cost,
)
from scripts.lib.validation.config.conventional import ConventionalConfig

SNAPSHOTS = pd.date_range("2023-01-01", periods=3, freq="h")


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", "DE00")
    n.add(
        "Generator",
        ["nuclear", "lignite"],
        bus="DE00",
        carrier=["nuclear", "lignite"],
        p_nom=[100.0, 200.0],
        p_min_pu=0.3,
        p_max_pu=pd.DataFrame(
            {"nuclear": [0.2, 0.5, 1.0], "lignite": [0.4, 0.5, 1.0]}, index=SNAPSHOTS
        ),
    )
    n.add(
        "Generator",
        "CCGT",
        bus="DE00",
        carrier="CCGT",
        p_nom=50.0,
        marginal_cost=pd.Series([80.0, 100.0, 120.0], index=SNAPSHOTS),
    )
    n.add(
        "Generator", "OCGT", bus="DE00", carrier="OCGT", p_nom=10.0, marginal_cost=150.0
    )
    return n


def test_clip_p_min_pu(network):
    clip_p_min_pu(network)

    p_min_pu = network.get_switchable_as_dense("Generator", "p_min_pu")
    np.testing.assert_allclose(p_min_pu.nuclear, [0.2, 0.3, 0.3])
    np.testing.assert_allclose(p_min_pu.lignite, 0.3)
    assert network.generators_t.p_min_pu.columns.tolist() == ["nuclear"]


@pytest.mark.parametrize(
    "spread, utilisation",
    [(0.3, 0.35), (0.5, 0.5), (0.0, 0.35)],
)
def test_spread_marginal_cost(network, spread, utilisation):
    spread_marginal_cost(network, ["CCGT", "OCGT"], spread, utilisation)

    g = network.generators.loc[["CCGT", "OCGT"]]
    linear = network.get_switchable_as_dense("Generator", "marginal_cost")[g.index]
    full = linear + 2 * g.marginal_cost_quadratic * g.p_nom
    mean = pd.Series({"CCGT": 100.0, "OCGT": 150.0})
    np.testing.assert_allclose(linear.mean(), mean * (1 - 2 * utilisation * spread))
    np.testing.assert_allclose(full.mean(), mean * (1 + 2 * (1 - utilisation) * spread))
    np.testing.assert_allclose(linear.CCGT.diff().dropna(), 20.0)
    assert network.generators.marginal_cost_quadratic.nuclear == 0.0


@pytest.mark.parametrize(
    "gas, factor, utilisation",
    [
        ([12.0, 24.0, 18.0], 0.4, 0.75),
        ([60.0, 90.0, 30.0], 1.0, 2 / 3),
        ([0.0, 0.0, 0.0], 1.0, 0.35),
        (None, 1.0, 0.35),
    ],
    ids=["below-p_nom", "above-p_nom", "zero-peak", "no-data"],
)
def test_spread_marginal_cost_observed(network, gas, factor, utilisation):
    observed = {("FR", "gas"): [1.0, 2.0, 3.0]} | (
        {("DE_LU", "gas"): gas} if gas else {}
    )
    generation = pd.DataFrame(observed, index=SNAPSHOTS)
    area_to_bus = pd.Series({"DE00": "DE_LU"})
    spread_marginal_cost(network, ["CCGT", "OCGT"], 0.3, 0.35, generation, area_to_bus)

    g = network.generators.loc[["CCGT", "OCGT"]]
    linear = network.get_switchable_as_dense("Generator", "marginal_cost")[g.index]
    mean = pd.Series({"CCGT": 100.0, "OCGT": 150.0})
    np.testing.assert_allclose(linear.mean(), mean * (1 - 2 * utilisation * 0.3))
    np.testing.assert_allclose(g.marginal_cost_quadratic * g.p_nom, 0.3 * mean / factor)


@pytest.mark.parametrize(
    "carriers, extendable, match",
    [
        (["CCGT", "oil"], False, "No generators"),
        (["CCGT"], True, "fixed positive p_nom"),
    ],
    ids=["missing-carrier", "extendable"],
)
def test_spread_marginal_cost_fails(network, carriers, extendable, match):
    network.generators.loc["CCGT", "p_nom_extendable"] = extendable
    with pytest.raises(AssertionError, match=match):
        spread_marginal_cost(network, carriers, 0.3, 0.35)


@pytest.mark.parametrize(
    "spread",
    [
        {"spread": -0.1},
        {"utilisation": 1.1},
        {"utilisation": -0.1},
        {"reference": "peak"},
    ],
)
def test_conventional_config_rejects_invalid_spread(spread):
    with pytest.raises(ValidationError):
        ConventionalConfig(marginal_cost_spread=spread)


DAYS = pd.date_range("2023-01-13", periods=3, freq="D")
CAP = {"2023-01-14": 50.0, "2023-01-15": 70.0}


@pytest.mark.parametrize(
    "cap, expected",
    [
        (CAP, [60.0, 50.0, 60.0]),
        ({"2023-01-01": 45.0}, [45.0, 45.0, 45.0]),
        ({"2023-02-01": 45.0}, [60.0, 60.0, 60.0]),
    ],
    ids=["steps", "before-snapshots", "after-snapshots"],
)
def test_cap_fuel_price(cap, expected):
    fuel_price = pd.DataFrame({"a": 60.0, "b": 40.0}, index=DAYS)
    result = cap_fuel_price(fuel_price, cap)
    np.testing.assert_allclose(result.a, expected)
    np.testing.assert_allclose(result.b, 40.0)


def test_gas_price_cap_before_spread():
    n = pypsa.Network()
    n.set_snapshots(DAYS)
    n.add("Bus", ["ES00", "FR00"], country=["ES", "FR"])
    costs = pd.DataFrame(
        {"fuel": 0.0, "VOM": 0.0, "CO2 intensity": 0.0}, index=["CCGT", "coal"]
    )
    ppl = pd.DataFrame(
        {
            "carrier": ["CCGT", "CCGT", "coal"],
            "bus": ["ES00", "FR00", "ES00"],
            "p_nom": 100.0,
            "efficiency": 0.5,
            "marginal_cost": 0.0,
            "capital_cost": 0.0,
            "build_year": 2000,
            "lifetime": 50.0,
        },
        index=["ES CCGT", "FR CCGT", "ES coal"],
    )
    fuel_price = pd.DataFrame({"gas": 60.0, "coal": 60.0}, index=DAYS)
    gas_cap = {"enable": True, "countries": ["ES"], "cap": CAP}
    attach_conventional_generators(
        n,
        costs,
        ppl,
        ["CCGT", "coal"],
        {"Generator": []},
        set(),
        {"gas_price_cap": gas_cap},
        {},
        fuel_price=fuel_price,
    )
    marginal_cost = n.generators_t.marginal_cost
    np.testing.assert_allclose(marginal_cost["ES CCGT"], [120.0, 100.0, 120.0])
    np.testing.assert_allclose(marginal_cost[["FR CCGT", "ES coal"]], 120.0)
    spread_marginal_cost(n, ["CCGT"], 0.3, 0.5)
    quadratic = n.generators.marginal_cost_quadratic
    np.testing.assert_allclose(quadratic["ES CCGT"] * 100.0 / 0.3, 340.0 / 3)


@pytest.mark.parametrize(
    "config",
    [
        {"gas_price_cap": {"enable": True, "cap": CAP}},
        {"gas_price_cap": {"enable": True, "countries": ["ES"]}},
        {"gas_price_cap": {"countries": ["ES"], "cap": {"January": 45.0}}},
        {"gas_price_cap": {"countries": ["ES"], "cap": {"2023-01-01": -1.0}}},
        {"gas_price_cap": {"enable": True, "countries": ["ES"], "cap": CAP}},
    ],
    ids=["no-countries", "no-cap", "invalid-date", "negative", "static-fuel-price"],
)
def test_conventional_config_rejects_invalid_gas_price_cap(config):
    with pytest.raises(ValidationError):
        ConventionalConfig(**config)


def test_conventional_config_accepts_gas_price_cap():
    gas_cap = {"enable": True, "countries": ["ES", "PT"], "cap": CAP}
    config = ConventionalConfig(dynamic_fuel_price=True, gas_price_cap=gas_cap)
    assert config.gas_price_cap.cap == CAP


@pytest.mark.parametrize(
    ("default_hub", "fr_cost"),
    [("THE", [80.0, 90.0, 100.0]), (None, 120.0)],
    ids=["default-hub", "monthly-fallback"],
)
def test_daily_gas_price_per_hub_before_cap(default_hub, fr_cost):
    n = pypsa.Network()
    n.set_snapshots(DAYS)
    n.add("Bus", ["ES00", "FR00"], country=["ES", "FR"])
    costs = pd.DataFrame(
        {"fuel": 0.0, "VOM": 0.0, "CO2 intensity": 0.0}, index=["CCGT", "coal"]
    )
    ppl = pd.DataFrame(
        {
            "carrier": ["CCGT", "CCGT", "coal"],
            "bus": ["ES00", "FR00", "ES00"],
            "p_nom": 100.0,
            "efficiency": 0.5,
            "marginal_cost": 0.0,
            "capital_cost": 0.0,
            "build_year": 2000,
            "lifetime": 50.0,
        },
        index=["ES CCGT", "FR CCGT", "ES coal"],
    )
    fuel_price = pd.DataFrame({"gas": 60.0, "coal": 60.0}, index=DAYS)
    gas_price = pd.DataFrame(
        {"THE": [40.0, 45.0, 50.0], "MIBGAS": [55.0, 52.0, 75.0]}, index=DAYS
    )
    conventional = {
        "gas_price_cap": {"enable": True, "countries": ["ES"], "cap": CAP},
        "daily_gas_price": {"default_hub": default_hub, "hubs": {"ES": "MIBGAS"}},
    }
    attach_conventional_generators(
        n,
        costs,
        ppl,
        ["CCGT", "coal"],
        {"Generator": []},
        set(),
        conventional,
        {},
        fuel_price=fuel_price,
        gas_price=gas_price,
    )
    marginal_cost = n.generators_t.marginal_cost
    np.testing.assert_allclose(marginal_cost["ES CCGT"], [110.0, 100.0, 140.0])
    np.testing.assert_allclose(marginal_cost["FR CCGT"], fr_cost)
    np.testing.assert_allclose(marginal_cost["ES coal"], 120.0)


@pytest.mark.parametrize(
    "config",
    [
        {"daily_gas_price": {"enable": True}},
        {"dynamic_fuel_price": True, "daily_gas_price": {"enable": True}},
        {"daily_gas_price": {"hubs": {"IT": "PSV"}}},
    ],
    ids=["static-fuel-price", "real-fuel-price", "unknown-hub"],
)
def test_conventional_config_rejects_invalid_daily_gas_price(config):
    with pytest.raises(ValidationError):
        ConventionalConfig(**config)


def test_conventional_config_accepts_daily_gas_price():
    daily = {"enable": True, "hubs": {"ES": "MIBGAS", "PT": "MIBGAS"}}
    config = ConventionalConfig(
        dynamic_fuel_price=True, fuel_price_nominal=True, daily_gas_price=daily
    )
    assert config.daily_gas_price.default_hub == "THE"
    daily["default_hub"] = None
    config = ConventionalConfig(
        dynamic_fuel_price=True, fuel_price_nominal=True, daily_gas_price=daily
    )
    assert config.daily_gas_price.default_hub is None
