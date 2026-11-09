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

from scripts.lib.backcast_calibration import read_solar_unpublished
from scripts.lib.backcast_dispatch import dispatch_carriers
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.validation.config import validate_config, validate_scenarios

ROOT = Path(__file__).parent.parent
BASE = yaml.safe_load((ROOT / "config/backcast/config.backcast.yaml").read_text())
SCENARIOS = yaml.safe_load((ROOT / BASE["run"]["scenarios"]["file"]).read_text())
DEFAULTS = yaml.safe_load((ROOT / "config/config.default.yaml").read_text())


def resolve(name: str) -> dict:
    """Return the defaults updated by the base config and the overrides of scenario `name`."""
    config = copy.deepcopy(DEFAULTS)
    update_config(config, BASE)
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
    defaults = copy.deepcopy(DEFAULTS)
    update_config(defaults, BASE)
    validate_scenarios(defaults, SCENARIOS)


def test_validates(scenario):
    _, _, config = scenario
    validate_config(config)


def test_year_settings_consistent(scenario):
    _, year, config = scenario
    electricity = config["electricity"]
    snapshots = config["snapshots"]
    assert (snapshots["start"], snapshots["end"]) == (
        f"{year}-01-01",
        f"{year + 1}-01-01",
    )
    assert config["atlite"]["default_cutout"] == f"europe-{year}-sarah3-era5"
    assert electricity["estimate_renewable_capacities"]["year"] == year
    assert config["conventional"]["estimate_efficiencies"]["reference_year"] == year
    for key in ["powerplants_filter", "custom_powerplants"]:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", electricity[key]))
        assert years == {str(year)}, f"electricity: {key} names years {years}"


def test_areas_known(scenario):
    _, _, config = scenario
    backcast = config["backcast"]
    area_to_bus = read_area_to_bus(ROOT / backcast["areas"]["area_to_bus"])
    borders = pd.read_csv(ROOT / backcast["areas"]["borders"])
    known = set(area_to_bus)
    assert known <= set(Area.__members__), "Areas unknown to entsoe-py"
    assert set(borders.area0) | set(borders.area1) <= known

    market_data = backcast["market_data"]
    wind = backcast["renewables"]["wind"]
    dispatch = backcast["dispatch"]
    configured = {
        "market_data: prices: exclude_areas": market_data["prices"]["exclude_areas"],
        "market_data: generation: exclude_areas": market_data["generation"][
            "exclude_areas"
        ],
        "interconnection: fixed": [
            a
            for border in backcast["interconnection"]["fixed"]
            for a in border.split("-")
        ],
        "renewables: wind: exclude": wind["exclude"],
        "renewables: wind: target_twh": wind["target_twh"],
        "renewables: wind: curtailment_twh": wind["curtailment_twh"],
        "dispatch: nuclear: phase_out": dispatch["nuclear"]["phase_out"],
        "dispatch: exclude_areas": [
            a
            for carrier, block in dispatch.items()
            if carrier != "chp_floor"
            for a in block["exclude_areas"]
        ],
    }
    for key, areas in configured.items():
        unknown = sorted(set(areas) - known)
        assert not unknown, (
            f"backcast: {key} names areas outside the area map: {unknown}"
        )


def test_external_borders_known(scenario):
    _, _, config = scenario
    area_to_bus = read_area_to_bus(ROOT / config["backcast"]["areas"]["area_to_bus"])
    modelled = set(area_to_bus[area_to_bus.index.str[:2].isin(config["countries"])])
    borders = pd.read_csv(ROOT / "data/entsoe_external_borders.csv")
    external = set(config["backcast"]["areas"]["external"]["areas"])
    assert set(borders.area) <= modelled
    assert external <= set(borders.external_area) <= set(Area.__members__)
    assert not external & modelled


@pytest.mark.parametrize(
    "path, off",
    [
        ("renewable.hydro.flatten_dispatch", False),
        ("renewable.hydro.inflow_source", "eia"),
        ("backcast.renewables.wind.calibrate", False),
        ("backcast.renewables.solar.calibrate", False),
        ("backcast.renewables.solar.unpublished", False),
        ("backcast.dispatch.nuclear.availability", "none"),
        ("backcast.dispatch.nuclear.min_output", "none"),
        ("backcast.dispatch.nuclear.fuel_price", 7.4536),
        ("backcast.dispatch.biomass.energy", "none"),
        ("backcast.dispatch.chp_floor.enable", False),
        ("backcast.interconnection.hybrid.enable", False),
        ("backcast.areas.external.enable", False),
    ],
)
def test_plain_scenario_unrefined(path, off):
    config = resolve("backcast-2019-plain")
    for key in path.split("."):
        config = config[key]
    assert config == off


def test_plain_scenario_without_dispatch_refinements():
    dispatch = dispatch_carriers(resolve("backcast-2019-plain")["backcast"]["dispatch"])
    assert not (dispatch.uses_monthly_generation or dispatch.uses_hourly_generation)
    assert not dispatch.outages


def test_solar_unpublished_calibrated(scenario):
    _, year, config = scenario
    backcast = config["backcast"]
    if not backcast["renewables"]["solar"]["unpublished"]:
        return
    areas = set(read_solar_unpublished(ROOT / "data/solar_unpublished.csv", year).index)
    assert areas <= set(read_area_to_bus(ROOT / backcast["areas"]["area_to_bus"]))
    excluded = (
        backcast["renewables"]["solar"]["exclude"]
        + backcast["market_data"]["generation"]["exclude_areas"]
    )
    assert not areas & set(excluded)


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


BIOMASS_TOPUP = pd.read_csv(ROOT / "test/test_data/backcast_biomass_topup.csv")


def test_biomass_topup_reaches_entsoe(scenario):
    _, year, config = scenario
    custom = pd.read_csv(ROOT / "data/custom_powerplants.csv").query(
        config["electricity"]["custom_powerplants"]
    )
    topup = custom.query("Name == 'Biomass units not in powerplantmatching'")
    assert (topup.Fueltype == "Bioenergy").all()
    added = topup.groupby("Country").Capacity.sum()
    expected = BIOMASS_TOPUP.query("year == @year").set_index("country")
    gap = expected.target_mw - expected.powerplantmatching_mw
    assert added.index.sort_values().equals(gap.index.sort_values())
    assert added[gap.index].sub(gap).between(0, 0.2).all()


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
