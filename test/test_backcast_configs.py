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
            a for areas in backcast.get("monthly_availability_exclude", {}).values() for a in areas
        ],
    }
    for key, areas in configured.items():
        unknown = sorted(set(areas) - known)
        assert not unknown, (
            f"backcast: {key} names areas outside the area map: {unknown}"
        )
