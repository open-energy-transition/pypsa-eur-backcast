# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds hourly observed day-ahead electricity prices per bus of the bidding zone clustered network.

The ENTSO-E prices per bidding zone are converted to UTC, resampled to hourly
means and restricted to the snapshots. Prices published in a local currency are
converted to EUR with the ECB euro reference rate of their UTC day, or the last
published rate on days without one (weekends and TARGET holidays). Gaps of up to
``MAX_INTERPOLATE_PRICE_GAP_HOURS`` are interpolated, gaps up to
``backcast: max_price_gap_hours`` (or the area specific value in
``backcast: max_price_gap_hours_by_area``) are kept as missing values and longer
gaps raise an error. The bidding zones are mapped to the
buses with ``backcast: area_to_bus``, where several buses may share one
bidding zone. Buses of areas listed in ``backcast: exclude_price_areas`` are
omitted.
"""

import logging

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.timeseries import fill_short_gaps
from scripts.lib.validation.config.backcast import MAX_INTERPOLATE_PRICE_GAP_HOURS

logger = logging.getLogger(__name__)


def to_hourly_utc(prices: pd.Series) -> pd.Series:
    """Convert timezone-aware prices to hourly means with naive UTC timestamps."""
    return prices.tz_convert("UTC").resample("h").mean().tz_localize(None)


def read_prices(path: str) -> tuple[pd.Series, str]:
    """Read the retrieved prices of one area and their currency, EUR for files retrieved before the currency was stored."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    currency = frame.pop("currency").iloc[0] if "currency" in frame else "EUR"
    return frame.squeeze("columns"), currency


def read_exchange_rates(path: str) -> pd.DataFrame:
    """Read the daily ECB reference rates in units of each currency per EUR."""
    rates = pd.read_csv(path, index_col="Date", parse_dates=True, na_values="N/A")
    return rates.sort_index()


def to_eur(prices: pd.Series, currency: str, rates: pd.DataFrame) -> pd.Series:
    """Convert hourly prices with naive UTC timestamps to EUR with the reference rate of their day or the last one before."""
    if currency == "EUR":
        return prices
    rate = rates[currency].dropna()
    days = prices.index.normalize()
    assert days.min() >= rate.index[0] and days.max() <= rate.index[-1], (
        f"No ECB {currency} rate for all days of {prices.name}"
    )
    return prices / rate.reindex(days, method="ffill").to_numpy()


def map_areas_to_buses(
    prices: pd.DataFrame,
    area_to_bus: pd.Series,
    buses: pd.Index,
    exclude: list[str],
    exclude_buses: list[str],
) -> pd.DataFrame:
    """Assign bidding zone prices to buses, omitting excluded areas and buses."""
    missing = buses.difference(area_to_bus.index)
    assert missing.empty, f"Buses without ENTSO-E area: {missing.tolist()}"
    unknown = prices.columns.difference(area_to_bus)
    assert unknown.empty, f"Unknown ENTSO-E areas: {unknown.tolist()}"
    bus_areas = area_to_bus.loc[buses[~buses.isin(exclude_buses)]]
    bus_areas = bus_areas[~bus_areas.isin(exclude)]
    return prices[bus_areas.to_numpy()].set_axis(bus_areas.index, axis=1)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_day_ahead_prices")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    buses = pypsa.Network(snakemake.input.network).buses.query("carrier == 'AC'").index

    rates = read_exchange_rates(snakemake.input.exchange_rates)
    raw = [read_prices(fn) for fn in snakemake.input.prices]
    for s, currency in raw:
        if currency != "EUR":
            logger.info(
                f"Converting day-ahead prices of {s.name} from {currency} to EUR."
            )
    hourly = pd.concat(
        [to_eur(to_hourly_utc(s), currency, rates) for s, currency in raw],
        keys=[s.name for s, _ in raw],
    ).unstack(level=0)

    prices = fill_short_gaps(
        hourly.reindex(snapshots),
        MAX_INTERPOLATE_PRICE_GAP_HOURS,
        snakemake.params.max_price_gap_hours,
        snakemake.params.max_price_gap_hours_by_area,
    )
    prices = map_areas_to_buses(
        prices,
        area_to_bus,
        buses,
        snakemake.params.exclude_price_areas,
        snakemake.params.exclude_price_buses,
    )

    prices.to_csv(snakemake.output.csv)
