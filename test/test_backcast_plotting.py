# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the data helpers of the backcast benchmark figures."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.lib.backcast_data import decile_errors, hourly_errors, net_exchanges
from scripts.lib.backcast_plotting import (
    area_flows,
    area_generation,
    ordered_areas,
    representative_buses,
    select_weeks,
)
from scripts.plot_backcast_system import area_net_export

AREA_TO_BUS = pd.Series(
    {"DE00": "DE_LU", "LUG1": "DE_LU", "FR00": "FR", "IE00": "IE_SEM", "GBNI": "IE_SEM"}
)
SNAPSHOTS = pd.date_range("2023-01-01", periods=3, freq="h")


@pytest.mark.parametrize(
    "observed, expected",
    [
        (["FR00", "LUG1", "DE00"], {"DE_LU": "DE00", "FR": "FR00"}),
        (["IE00", "GBNI", "LUG1"], {"DE_LU": "LUG1", "IE_SEM": "GBNI"}),
    ],
)
def test_representative_buses(observed, expected):
    result = representative_buses(AREA_TO_BUS, pd.Index(observed))

    assert result.to_dict() == expected
    assert list(result.index) == list(expected)


def test_ordered_areas():
    assert ordered_areas(["IE_SEM", "ES", "DE_LU"]) == ["DE_LU", "ES", "IE_SEM"]


def test_ordered_areas_rejects_unknown_area():
    with pytest.raises(AssertionError, match="ZONE_GROUPS"):
        ordered_areas(["XX"])


@pytest.mark.parametrize(
    "low, high, winter, summer",
    [
        ("2023-07-02 12:00", "2023-01-25 12:00", "2023-01-23", "2023-06-26"),
        ("2023-04-03 00:00", "2023-12-27 00:00", "2023-12-25", "2023-04-03"),
    ],
)
def test_select_weeks(low, high, winter, summer):
    index = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
    prices = pd.Series(50.0, index)
    prices[low] = -500.0
    prices[high] = 5000.0

    weeks = select_weeks(prices)

    assert weeks["winter"][0] == pd.Timestamp(winter)
    assert weeks["summer"][0] == pd.Timestamp(summer)
    assert all(len(week) == 168 for week in weeks.values())


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    n.add(
        "Link",
        ["FR00->DE00", "DE00->FR00", "LUG1->FR00", "copper"],
        bus0=["FR00", "DE00", "LUG1", "DE00"],
        bus1=["DE00", "FR00", "FR00", "LUG1"],
        carrier=["DC", "DC", "DC", "copper"],
        p_nom=1.0,
    )
    n.links_t.p0 = pd.DataFrame(
        {"FR00->DE00": [1.0, 0, 2], "DE00->FR00": [0.0, 3, 0], "LUG1->FR00": 1.0},
        SNAPSHOTS,
    )
    n.add("Generator", ["DE00 CCGT", "LUG1 OCGT"], bus=["DE00", "LUG1"])
    n.generators.carrier = ["CCGT", "OCGT"]
    n.add("StorageUnit", "FR00 PHS", bus="FR00", carrier="PHS")
    n.generators_t.p = pd.DataFrame({"DE00 CCGT": 1.0, "LUG1 OCGT": 2.0}, SNAPSHOTS)
    n.storage_units_t.p = pd.DataFrame({"FR00 PHS": [1.0, -2, 3]}, SNAPSHOTS)
    return n


def test_area_flows_nets_directions_and_skips_intra_area(network):
    flows = area_flows(network, AREA_TO_BUS)

    assert flows.columns.tolist() == ["DE_LU->FR"]
    np.testing.assert_allclose(flows["DE_LU->FR"], [0.0, 4, -1])


def test_area_generation_groups_carriers_and_positive_storage(network):
    generation = area_generation(network, AREA_TO_BUS)

    np.testing.assert_allclose(generation["DE_LU", "gas"], 3.0)
    np.testing.assert_allclose(generation["FR", "PHS"], [1.0, 0, 3])


@pytest.mark.parametrize(
    "backward, expected",
    [([1.0, 1, 1, 1], ["A->B"]), ([np.nan, np.nan, 1, 1], [])],
)
def test_net_exchanges_requires_coverage(backward, expected):
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    exchanges = pd.DataFrame({"A->B": [3.0, 3, 3, 3], "B->A": backward}, index)

    net = net_exchanges(exchanges)

    assert net.columns.tolist() == expected
    if expected:
        np.testing.assert_allclose(net["A->B"], 2.0)


def test_area_net_export():
    flows = pd.DataFrame({"A->B": [1e6, 1e6], "B->C": [-1e6, 0.0]})

    export = area_net_export(flows, ["A", "B", "C"])

    assert export.to_dict() == {"A": 2.0, "B": -3.0, "C": 1.0}


@pytest.mark.parametrize(
    "error, months, expected",
    [
        ([1.0, 2, 3], [1], [2.0, 2.0]),
        ([1.0, -1, 4], [1], [2.5, -1.0]),
    ],
)
def test_hourly_errors_use_cet(error, months, expected):
    index = pd.DatetimeIndex(
        ["2023-01-01 23:00", "2023-01-02 00:00", "2023-01-02 23:00"]
    )
    result = hourly_errors(pd.DataFrame({"X": error}, index), months)

    assert result.columns.tolist() == list(range(len(expected)))
    np.testing.assert_allclose(result.loc["X"], expected)


def test_decile_errors_group_by_observed_rank():
    observed = pd.DataFrame({"X": np.arange(20.0)})
    error = pd.DataFrame({"X": np.repeat(np.arange(10.0), 2)})

    result = decile_errors(error, observed)

    np.testing.assert_allclose(result.loc["X"], np.arange(10.0))
