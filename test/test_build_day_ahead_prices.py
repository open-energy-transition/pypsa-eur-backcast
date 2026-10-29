# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for build_day_ahead_prices.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from entsoe.mappings import Area

from scripts.build_day_ahead_prices import (
    map_areas_to_buses,
    read_exchange_rates,
    read_prices,
    to_eur,
    to_hourly_utc,
)
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.timeseries import fill_short_gaps


@pytest.fixture
def area_to_bus():
    path = Path(__file__).parent.parent / "data" / "entsoe_area_to_bus.csv"
    return read_area_to_bus(path)


@pytest.fixture
def prices():
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    return pd.DataFrame({"DE_LU": [1.0, 2, 3], "IE_SEM": [4.0, 5, 6]}, index=index)


def test_area_to_bus_uses_entsoe_areas(area_to_bus):
    assert area_to_bus.index.is_unique
    assert set(area_to_bus) <= {a.name for a in Area}


@pytest.mark.parametrize(
    "start, utc_start",
    [
        ("2023-03-26 00:00", "2023-03-25 23:00"),
        ("2023-10-29 00:00", "2023-10-28 22:00"),
    ],
    ids=["spring", "autumn"],
)
def test_to_hourly_utc_across_dst(start, utc_start):
    index = pd.date_range(start, periods=16, freq="15min", tz="Europe/Brussels")
    prices = pd.Series(np.repeat([1.0, 2, 3, 4], 4) + np.tile([0, 1, 2, 3], 4), index)

    result = to_hourly_utc(prices)

    expected_index = pd.date_range(utc_start, periods=4, freq="h")
    pd.testing.assert_series_equal(
        result, pd.Series([2.5, 3.5, 4.5, 5.5], expected_index), check_freq=False
    )


def test_map_areas_to_buses_many_to_one(prices, area_to_bus):
    buses = pd.Index(["DE00", "LUG1", "IE00", "GBNI"])

    result = map_areas_to_buses(
        prices, area_to_bus, buses, exclude=[], exclude_buses=[]
    )

    assert result.columns.equals(buses)
    np.testing.assert_array_equal(result["LUG1"], prices["DE_LU"])
    np.testing.assert_array_equal(result["GBNI"], prices["IE_SEM"])


@pytest.mark.parametrize(
    "exclude, exclude_buses",
    [(["GB"], []), ([], ["GB00"])],
    ids=["area", "bus"],
)
def test_map_areas_to_buses_exclude(prices, area_to_bus, exclude, exclude_buses):
    buses = pd.Index(["DE00", "GB00"])
    prices = prices.assign(GB=1.0)

    result = map_areas_to_buses(prices, area_to_bus, buses, exclude, exclude_buses)

    assert result.columns.tolist() == ["DE00"]


@pytest.mark.parametrize(
    "buses, columns, error",
    [
        (["DE00", "XX00"], ["DE_LU"], AssertionError),
        (["DE00"], ["DE_LU", "XX"], AssertionError),
        (["DE00", "GB00"], ["DE_LU"], KeyError),
    ],
    ids=["unmapped-bus", "unknown-area", "missing-area-data"],
)
def test_map_areas_to_buses_raises(area_to_bus, buses, columns, error):
    prices = pd.DataFrame(1.0, index=range(2), columns=columns)
    with pytest.raises(error):
        map_areas_to_buses(
            prices, area_to_bus, pd.Index(buses), exclude=[], exclude_buses=[]
        )


@pytest.mark.parametrize(
    "values, by_area, expected",
    [
        ([np.nan, 1, np.nan, np.nan, 4, np.nan], {}, [1.0, 1, 2, 3, 4, 4]),
        (
            [1, np.nan, np.nan, np.nan, np.nan, 6],
            {},
            [1.0, np.nan, np.nan, np.nan, np.nan, 6],
        ),
        ([1, np.nan, np.nan, np.nan, np.nan, np.nan], {}, None),
        (
            [1, np.nan, np.nan, np.nan, np.nan, np.nan],
            {"DE_LU": 5, "IE_SEM": 3},
            [1.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        ),
        ([1, np.nan, np.nan, np.nan, np.nan, 6], {"DE_LU": 3}, None),
    ],
    ids=["interpolate", "keep", "raise", "area-tolerate", "area-raise"],
)
def test_fill_short_gaps(values, by_area, expected):
    prices = pd.DataFrame({"DE_LU": values}, dtype=float)
    if expected is None:
        with pytest.raises(AssertionError, match="Gaps longer"):
            fill_short_gaps(
                prices, max_interpolate=3, max_gap=4, max_gap_by_column=by_area
            )
    else:
        result = fill_short_gaps(
            prices, max_interpolate=3, max_gap=4, max_gap_by_column=by_area
        )
        np.testing.assert_array_equal(result["DE_LU"], expected)


@pytest.fixture
def rates(tmp_path):
    path = tmp_path / "eurofxref-hist.csv"
    path.write_text(
        "Date,PLN,BGN,\n"
        "2019-01-07,4.0,1.95583,\n"
        "2019-01-04,5.0,1.95583,\n"
        "2019-01-03,N/A,1.95583,\n"
    )
    return read_exchange_rates(path)


def test_to_eur_uses_rate_of_day_or_last_before(rates):
    index = pd.date_range("2019-01-04 22:00", "2019-01-07 01:00", freq="h")
    prices = pd.Series(100.0, index, name="PL")

    result = to_eur(prices, "PLN", rates)

    expected = np.where(index < "2019-01-07", 100 / 5.0, 100 / 4.0)
    np.testing.assert_allclose(result, expected)
    assert result.name == "PL"


def test_to_eur_keeps_eur_and_raises_without_rate(rates):
    prices = pd.Series(1.0, pd.date_range("2019-01-03", periods=2, freq="h"))

    assert to_eur(prices, "EUR", rates) is prices
    with pytest.raises(AssertionError, match="No ECB PLN rate"):
        to_eur(prices, "PLN", rates)
    np.testing.assert_allclose(to_eur(prices, "BGN", rates), 1 / 1.95583)


@pytest.mark.parametrize(
    "content, currency",
    [
        ("utc_timestamp,PL,currency\n2019-01-01 00:00:00+00:00,200.0,PLN\n", "PLN"),
        ("utc_timestamp,DE_LU\n2019-01-01 00:00:00+00:00,50.0\n", "EUR"),
    ],
    ids=["currency", "legacy"],
)
def test_read_prices(tmp_path, content, currency):
    path = tmp_path / "prices.csv"
    path.write_text(content)

    prices, result = read_prices(path)

    assert result == currency and isinstance(prices, pd.Series)
    assert prices.index.tz is not None
