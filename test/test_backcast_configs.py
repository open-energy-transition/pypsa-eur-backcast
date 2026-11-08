# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the backcast base config and its year scenarios in config/backcast."""

import copy
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from entsoe.mappings import Area
from snakemake.utils import update_config

from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.validation.config import validate_config, validate_scenarios

ROOT = Path(__file__).parent.parent
BASE = yaml.safe_load((ROOT / "config/backcast/config.backcast.yaml").read_text())
SCENARIOS = yaml.safe_load((ROOT / BASE["run"]["scenarios"]["file"]).read_text())


def resolve(name: str, base: dict = BASE) -> dict:
    """Return `base` updated by the overrides of scenario `name`."""
    config = copy.deepcopy(base)
    update_config(config, SCENARIOS[name])
    return config


@pytest.fixture(params=list(SCENARIOS))
def scenario(request) -> tuple[str, int, dict]:
    name = request.param
    return name, int(name.split("-")[1]), resolve(name)


def test_scenarios_found():
    assert SCENARIOS, "No backcast scenarios in config/backcast/scenarios.yaml"
    assert all(re.fullmatch(r"backcast-\d{4}(-[a-z]+)?", name) for name in SCENARIOS)


def test_scenarios_compatible():
    """Each scenario resolves to a valid config without changing the planning horizon, as checked by the workflow."""
    defaults = yaml.safe_load((ROOT / "config/config.default.yaml").read_text())
    update_config(defaults, BASE)
    validate_scenarios(defaults, SCENARIOS)


def test_validates(scenario):
    _, _, config = scenario
    validate_config(config)


def test_year_settings_consistent(scenario):
    _, year, config = scenario
    electricity = config["electricity"]
    assert config["snapshots"] == {"start": f"{year}-01-01", "end": f"{year + 1}-01-01"}
    assert config["atlite"]["default_cutout"] == f"europe-{year}-sarah3-era5"
    assert electricity["estimate_renewable_capacities"]["year"] == year
    assert config["conventional"]["estimate_efficiencies"]["reference_year"] == year
    for key in ["powerplants_filter", "custom_powerplants"]:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", electricity[key]))
        assert years == {str(year)}, f"electricity: {key} names years {years}"


def test_areas_known(scenario):
    _, _, config = scenario
    backcast = config["backcast"]
    area_to_bus = read_area_to_bus(
        ROOT / backcast.get("area_to_bus", "data/entsoe_area_to_bus.csv")
    )
    borders = pd.read_csv(ROOT / backcast.get("borders", "data/entsoe_borders.csv"))
    known = set(area_to_bus)
    assert known <= set(Area.__members__), "Areas unknown to entsoe-py"
    assert set(borders.area0) | set(borders.area1) <= known

    fixed = [
        a
        for border in backcast.get("fixed_border_capacities", {})
        for a in border.split("-")
    ]
    configured = {
        "exclude_price_areas": backcast.get("exclude_price_areas", []),
        "exclude_generation_areas": backcast.get("exclude_generation_areas", []),
        "fixed_border_capacities": fixed,
        "wind_calibration_exclude": backcast.get("wind_calibration_exclude", {}),
        "wind_target_twh": backcast.get("wind_target_twh", {}),
        "wind_curtailment_twh": backcast.get("wind_curtailment_twh", {}),
        "nuclear_phase_out": backcast.get("nuclear_phase_out", {}),
        "monthly_availability_exclude": [
            a
            for areas in backcast.get("monthly_availability_exclude", {}).values()
            for a in areas
        ],
    }
    for key, areas in configured.items():
        unknown = sorted(set(areas) - known)
        assert not unknown, (
            f"backcast: {key} names areas outside the area map: {unknown}"
        )


def test_external_borders_known(scenario):
    _, _, config = scenario
    defaults = yaml.safe_load((ROOT / "config/config.default.yaml").read_text())
    update_config(defaults, config)
    area_to_bus = read_area_to_bus(ROOT / defaults["backcast"]["area_to_bus"])
    modelled = set(area_to_bus[area_to_bus.index.str[:2].isin(defaults["countries"])])
    borders = pd.read_csv(ROOT / "data/entsoe_external_borders.csv")
    external = set(defaults["backcast"]["external_flows"]["areas"])
    assert set(borders.area) <= modelled
    assert external <= set(borders.external_area) <= set(Area.__members__)
    assert not external & modelled


@pytest.mark.parametrize(
    "path, off",
    [
        ("renewable.hydro.flatten_dispatch", False),
        ("renewable.hydro.inflow_source", "eia"),
        ("backcast.calibrate_wind", False),
        ("backcast.calibrate_solar", False),
        ("backcast.monthly_nuclear_availability", False),
        ("backcast.nuclear_min_output", False),
        ("backcast.energy_budget_carriers", []),
        ("backcast.chp_floor.enable", False),
        ("backcast.hybrid_capacities.enable", False),
        ("backcast.external_flows.enable", False),
        ("backcast.fuel_price", {"nuclear": 7.4536, "biomass": 9.3506}),
    ],
)
def test_plain_scenario_unrefined(path, off):
    config = resolve("backcast-2019-plain")
    for key in path.split("."):
        config = config[key]
    assert config == off


ENTSOE_INSTALLED = pd.read_csv(ROOT / "test/test_data/backcast_installed_capacity.csv")


def test_custom_powerplants_match_entsoe(scenario):
    _, year, config = scenario
    custom = pd.read_csv(ROOT / "data/custom_powerplants.csv").query(
        config["electricity"]["custom_powerplants"]
    )
    assert custom.Source.notna().all()
    fleet = custom.groupby(["Country", "Fueltype"]).Capacity.sum()
    reference = ENTSOE_INSTALLED.query("year == @year")
    assert not reference.empty
    for row in reference.itertuples():
        key = (row.country, row.fueltype)
        assert fleet[key] == pytest.approx(row.capacity_mw, abs=150), key


def test_lignite_p_min_pu_covers_countries(scenario):
    _, year, config = scenario
    p_min_pu = config["conventional"]["lignite"]["p_min_pu"]
    if not isinstance(p_min_pu, str):
        return
    table = pd.read_csv(ROOT / p_min_pu, index_col=0)
    assert str(year) in table.columns
    countries = yaml.safe_load((ROOT / "config/config.default.yaml").read_text())[
        "countries"
    ]
    assert set(countries) <= set(table.index)
    assert table[str(year)].between(0, 1).all()


ALL_YEARS = range(2019, 2026)


@pytest.mark.parametrize(
    "country,name,fueltype,dropped",
    [
        ("FI", "Olkiluoto 1", "Nuclear", ALL_YEARS),
        ("FI", "Alholmens Kraft", "Bioenergy", ALL_YEARS),
        ("FI", "Kemi Bioproduct Mill", "Bioenergy", range(2023, 2026)),
        ("FI", "Kymijarvi", "Bioenergy", range(0)),
        ("FI", "Haapavesi", "Lignite", ALL_YEARS),
        ("FI", "Naantali Chp", "Hard Coal", range(2021, 2026)),
        ("FI", "Meri-Pori", "Hard Coal", range(0)),
        ("BA", "Capljina", "Hydro", ALL_YEARS),
        ("BA", "Tuzla", "Lignite", ALL_YEARS),
        ("BA", "Stanari", "Lignite", range(0)),
        ("EE", "Kiisa Erpp", "Oil", ALL_YEARS),
        ("EE", "Iru Elektrijaam", "Natural Gas", ALL_YEARS),
        ("SE", "Stenungsund", "Oil", ALL_YEARS),
        ("SE", "Oresundsverket", "Oil", range(2019, 2025)),
        ("SE", "Karlshamn", "Oil", range(0)),
    ],
)
def test_powerplants_filter_replaces_fleet(scenario, country, name, fueltype, dropped):
    _, year, config = scenario
    ppl = pd.DataFrame(
        {
            "Country": [country],
            "Name": [name],
            "Fueltype": [fueltype],
            "DateIn": [1990.0],
            "DateOut": [float("nan")],
        }
    )
    kept = len(ppl.query(config["electricity"]["powerplants_filter"])) == 1
    assert kept == (year not in dropped)
