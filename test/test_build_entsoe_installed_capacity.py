# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from scripts.build_entsoe_installed_capacity import (
    entsoe_capacity,
    fleet_audit,
    model_capacity,
    reporting_area,
)


@pytest.fixture
def raw() -> dict[str, pd.DataFrame]:
    return {
        "EE": pd.DataFrame({"Fossil Oil shale": [2000.0], "Fossil Gas": [300.0]}),
        "FI": pd.DataFrame(
            {
                "Fossil Peat": [1000.0],
                "Fossil Brown coal/Lignite": [50.0],
                "Other": [9.0],
            }
        ),
        "XX": pd.DataFrame(),
    }


@pytest.mark.parametrize(
    "area, carrier, expected",
    [("EE", "lignite", 2000.0), ("EE", "gas", 300.0), ("FI", "lignite", 1050.0)],
)
def test_entsoe_capacity_maps_and_sums_production_types(raw, area, carrier, expected):
    result = entsoe_capacity(raw)
    assert result.loc[(area, carrier), "entsoe_mw"] == expected
    assert ("FI", "Other") not in result.index
    assert "XX" not in result.index.get_level_values("area")


def test_model_capacity_sums_buses_and_carriers():
    p_nom = pd.DataFrame(
        {
            "bus": ["IT00", "IT01", "IT01", "EE00"],
            "carrier": ["CCGT", "OCGT", "solar-hsat", "oil"],
            "p_nom": [100.0, 20.0, 5.0, 7.0],
        }
    )
    area_to_bus = pd.Series(
        {"IT00": "IT_N", "IT01": "IT_N", "EE00": "EE"}, name="entsoe_area"
    )
    result = model_capacity(p_nom, area_to_bus)
    assert result.loc[("IT_N", "gas")] == 120.0
    assert result.loc[("IT_N", "solar")] == 5.0
    assert result.loc[("EE", "oil")] == 7.0


def test_model_capacity_rejects_unmapped_bus():
    p_nom = pd.DataFrame({"bus": ["ZZ00"], "carrier": ["oil"], "p_nom": [1.0]})
    with pytest.raises(AssertionError):
        model_capacity(p_nom, pd.Series({"EE00": "EE"}))


def test_fleet_audit_gap_is_model_minus_entsoe(raw):
    model = pd.Series(
        {("EE", "gas"): 250.0, ("EE", "oil"): 251.0}, name="model_mw"
    ).rename_axis(["area", "carrier"])
    fleet = fleet_audit(model, entsoe_capacity(raw)).set_index(["area", "carrier"])
    assert fleet.loc[("EE", "gas"), "gap_mw"] == -50.0
    assert fleet.loc[("EE", "lignite"), "gap_mw"] == -2000.0
    assert fleet.loc[("EE", "oil"), "gap_mw"] == 251.0


def test_fleet_audit_without_model_has_no_gap(raw):
    fleet = fleet_audit(None, entsoe_capacity(raw))
    assert fleet.model_mw.isna().all()
    assert np.isnan(fleet.gap_mw).all()


@pytest.mark.parametrize(
    "area, expected",
    [
        ("IT_NORD", "IT"),
        ("SE_3", "SE"),
        ("NO_2", "NO_2"),
        ("DE_LU", "DE_LU"),
        ("FR", "FR"),
    ],
)
def test_reporting_area_merges_italian_and_swedish_zones(area, expected):
    assert reporting_area(area) == expected
