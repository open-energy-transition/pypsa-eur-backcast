# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the carrier lists derived from the backcast dispatch blocks."""

import pytest

from scripts.lib.backcast_dispatch import dispatch_carriers
from scripts.lib.validation.config.backcast_dispatch import DispatchConfig


def carriers(**blocks: dict):
    return dispatch_carriers(DispatchConfig(**blocks).model_dump())


@pytest.mark.parametrize(
    "blocks, field, expected",
    [
        ({}, "monthly", []),
        ({}, "outages", []),
        ({}, "budget", []),
        ({}, "nuclear_monthly", False),
        ({"nuclear": {"availability": "monthly"}}, "nuclear_monthly", True),
        ({"nuclear": {"availability": "outages"}}, "nuclear_monthly", True),
        ({"nuclear": {"availability": "outages"}}, "outages", ["nuclear"]),
        ({"nuclear": {"availability": "monthly"}}, "monthly", []),
        ({"nuclear": {"min_output": "monthly_min"}}, "nuclear_min_output", True),
        (
            {
                c: {"availability": "monthly"}
                for c in ["waste", "biomass", "lignite", "coal"]
            },
            "monthly",
            ["coal", "lignite", "biomass", "waste"],
        ),
        (
            {c: {"availability": "outages"} for c in ["oil", "CCGT"]},
            "outages",
            ["CCGT", "oil"],
        ),
        ({"biomass": {"energy": "budget"}}, "budget", ["biomass"]),
        ({"biomass": {"energy": "budget"}}, "eurostat", []),
        ({"biomass": {"energy": "eurostat_budget"}}, "eurostat", ["biomass"]),
        ({"biomass": {"energy": "eurostat_budget"}}, "fixed", []),
        ({"biomass": {"energy": "fixed"}}, "budget", ["biomass"]),
        ({"biomass": {"energy": "fixed"}}, "eurostat", ["biomass"]),
        ({"biomass": {"energy": "fixed"}}, "fixed", ["biomass"]),
        ({"nuclear": {"fuel_price": 1.31}}, "fuel_price", {"nuclear": 1.31}),
    ],
)
def test_dispatch_carriers(blocks, field, expected):
    assert carriers(**blocks)._asdict()[field] == expected


def test_dispatch_carriers_per_carrier_settings():
    dispatch = carriers(
        biomass={"exclude_areas": ["BE"], "monthly_ceiling": False},
        coal={"outage_min_coverage": 0.5},
        nuclear={"availability": "outages"},
    )
    assert dispatch.exclude["biomass"] == ["BE"]
    assert dispatch.exclude["nuclear"] == []
    assert not dispatch.monthly_ceiling["biomass"]
    assert dispatch.min_coverage["nuclear"] == 0.9
    assert dispatch.min_coverage["coal"] == 0.5
    assert "chp_floor" not in dispatch.exclude


@pytest.mark.parametrize(
    "blocks, monthly, hourly",
    [
        ({}, False, False),
        ({"nuclear": {"availability": "monthly"}}, True, False),
        ({"coal": {"availability": "monthly"}}, True, False),
        ({"nuclear": {"min_output": "monthly_min"}}, False, True),
        ({"biomass": {"energy": "budget"}}, False, True),
    ],
)
def test_dispatch_carriers_generation_inputs(blocks, monthly, hourly):
    dispatch = carriers(**blocks)
    assert (dispatch.uses_monthly_generation, dispatch.uses_hourly_generation) == (
        monthly,
        hourly,
    )
