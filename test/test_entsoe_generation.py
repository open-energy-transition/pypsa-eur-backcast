# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the ENTSO-E actual generation retrieval and build."""

import numpy as np
import pandas as pd

from scripts.build_entsoe_generation import (
    hourly_carrier_generation,
    hourly_generation,
    monthly_generation,
)
from scripts.retrieve_actual_generation_entsoe import aggregated_generation
from test.backcast_calibration_network import SNAPSHOTS, frame


def test_aggregated_generation_drops_consumption_and_converts_to_utc():
    index = pd.date_range("2023-12-31 23:00", periods=3, freq="h", tz="Europe/Berlin")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Solar", "Actual Aggregated"),
            ("Solar", "Actual Consumption"),
            ("Nuclear", "Actual Aggregated"),
        ]
    )
    raw = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=index, columns=columns)
    result = aggregated_generation(raw, pd.Timestamp("2024-01-01", tz="UTC"))
    assert result.columns.tolist() == ["Solar", "Nuclear"]
    assert result.index.equals(index[:2].tz_convert("UTC"))


def test_aggregated_generation_merges_mixed_periods():
    plain = pd.DataFrame(
        {"Nuclear": [1.0, 2.0], "Solar": [3.0, 4.0]},
        index=pd.date_range("2020-09-30 22:00", periods=2, freq="h", tz="UTC"),
    )
    nested = pd.DataFrame(
        [[5.0, 6.0, 0.5]],
        index=pd.date_range("2020-10-01 00:00", periods=1, freq="h", tz="UTC"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Nuclear", "Actual Aggregated"),
                ("Solar", "Actual Aggregated"),
                ("Solar", "Actual Consumption"),
            ]
        ),
    )
    raw = pd.concat([plain, nested])
    result = aggregated_generation(raw, pd.Timestamp("2021-01-01", tz="UTC"))
    assert result.columns.tolist() == ["Nuclear", "Solar"]
    np.testing.assert_allclose(result.Nuclear, [1, 2, 5])
    np.testing.assert_allclose(result.Solar, [3, 4, 6])


def test_hourly_carrier_generation_maps_production_types():
    index = pd.date_range("2023-01-01", periods=8, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Fossil Brown coal/Lignite": [4.0] * 8,
            "Fossil Oil shale": [np.nan] * 4 + [2.0] * 4,
            "Wind Offshore": [1.0] * 8,
            "Other": [9.0] * 8,
        },
        index=index,
    )
    result = hourly_carrier_generation(raw)
    expected = pd.DataFrame(
        {"lignite": [4.0, 6.0], "offwind": [1.0, 1.0]},
        index=pd.date_range("2023-01-01", periods=2, freq="h"),
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_monthly_generation_sums_within_snapshots():
    hourly = pd.DataFrame(
        {
            "nuclear": [1.0, 2.0, np.nan, 4.0, 8.0],
            "onwind": [5.0, np.nan, np.nan, 3.0, np.nan],
        },
        index=pd.date_range("2023-01-31 21:00", periods=5, freq="h"),
    )
    result = monthly_generation(hourly_generation({"ES": hourly}, SNAPSHOTS))
    expected = frame(
        {
            ("ES", "nuclear", "2023-01-01"): (2.0, 1),
            ("ES", "nuclear", "2023-02-01"): (12.0, 2),
            ("ES", "onwind", "2023-02-01"): (3.0, 1),
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)
