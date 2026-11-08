# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for querying the ENTSO-E Transparency Platform and mapping its bidding zone areas to model buses.
"""

import os
from collections.abc import Iterable

import pandas as pd
from entsoe import EntsoePandasClient


def entsoe_client() -> EntsoePandasClient:
    """Create an ENTSO-E client with the API token from ``ENTSOE_API_TOKEN``."""
    token = os.environ["ENTSOE_API_TOKEN"]
    assert token, "Environment variable ENTSOE_API_TOKEN is empty."
    return EntsoePandasClient(api_key=token, timeout=120)


def utc_year(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start and end of a calendar year in UTC."""
    return pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(
        f"{year + 1}-01-01", tz="UTC"
    )


def to_utc(series: pd.Series, end: pd.Timestamp, name: str) -> pd.Series:
    """Convert a queried series to UTC timestamps before `end`."""
    result = series.tz_convert("UTC").loc[lambda s: s.index < end].rename(name)
    result.index.name = "utc_timestamp"
    return result


def read_area_to_bus(path: str) -> pd.Series:
    """Read the many-to-one map from model buses to ENTSO-E bidding zone areas."""
    return pd.read_csv(path, index_col="bus")["entsoe_area"]


def read_hourly_generation(path: str) -> pd.DataFrame:
    """Read the hourly actual generation in MW with area and carrier columns."""
    return pd.read_csv(path, index_col=0, header=[0, 1], parse_dates=True)


def assert_known_areas(areas: Iterable[str], area_to_bus: pd.Series, key: str) -> None:
    """Assert that the configured `areas` of `key` are ENTSO-E areas of the bus map."""
    unknown = sorted(set(areas) - set(area_to_bus))
    assert not unknown, f"Unknown ENTSO-E areas in {key}: {unknown}"
