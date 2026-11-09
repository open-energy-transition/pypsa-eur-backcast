# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds hourly observed day-ahead electricity prices per bus of the bidding zone clustered network.

The ENTSO-E prices per bidding zone are converted to UTC, resampled to hourly
means and restricted to the snapshots. Prices published in a local currency are
converted to EUR per price, as the currency may change within a year (RO from
RON to EUR in June 2021, BG from BGN to EUR in January 2022), with the ECB euro
reference rate of their UTC day, or the last published rate on days without one
(weekends and TARGET holidays). BGN is converted at its fixed peg of 1.95583 per
EUR, which the ECB publishes rounded to 1.9558. Gaps of up to
``MAX_INTERPOLATE_PRICE_GAP_HOURS`` are interpolated, gaps up to
``backcast: market_data: prices: max_gap_hours`` (or the area specific value in
``backcast: market_data: prices: max_gap_hours_by_area``) are kept as missing values and longer
gaps raise an error. The bidding zones are mapped to the
buses with ``backcast: areas: area_to_bus``, where several buses may share one
bidding zone. Buses of areas listed in ``backcast: market_data: prices: exclude_areas`` are
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

EUR_PEGS = {"BGN": 1.95583}


def to_hourly_utc(prices: pd.Series) -> pd.Series:
    """Convert timezone-aware prices to hourly means with naive UTC timestamps."""
    return prices.tz_convert("UTC").resample("h").mean().tz_localize(None)


def read_prices(path: str) -> tuple[pd.Series, pd.Series]:
    """Read the retrieved prices of one area and the currency of each price."""
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    assert "currency" in frame, f"No currency column in {path}, retrieve it again"
    currency = frame.pop("currency")
    return frame.squeeze("columns"), currency


def read_exchange_rates(path: str) -> pd.DataFrame:
    """Read the daily ECB reference rates in units of each currency per EUR, with pegged currencies at their unrounded peg."""
    rates = pd.read_csv(
        path, index_col="Date", parse_dates=True, na_values="N/A"
    ).sort_index()
    pegs = {c: rates[c].mask(rates[c].notna(), peg) for c, peg in EUR_PEGS.items()}
    return rates.assign(**pegs)


def daily_rate(days: pd.Series, currency: str, rates: pd.DataFrame) -> pd.Series:
    """Return the reference rate of each day or the last one before, 1 for EUR."""
    if currency == "EUR":
        return pd.Series(1.0, days.index)
    rate = rates[currency].dropna()
    assert days.min() >= rate.index[0] and days.max() <= rate.index[-1], (
        f"No ECB {currency} rate for all days of {days.name}"
    )
    return pd.Series(rate.reindex(days, method="ffill").to_numpy(), days.index)


def to_eur(prices: pd.Series, currency: pd.Series, rates: pd.DataFrame) -> pd.Series:
    """Convert timezone-aware prices to EUR with the reference rate of the UTC day of each price or the last one before."""
    foreign = sorted(set(currency) - {"EUR"})
    if foreign:
        logger.info(
            f"Converting day-ahead prices of {prices.name} from {foreign} to EUR."
        )
    utc_days = prices.index.tz_convert("UTC").tz_localize(None).normalize()
    days = pd.Series(utc_days, prices.index, name=prices.name)
    rate = pd.concat(
        [daily_rate(group, code, rates) for code, group in days.groupby(currency)]
    )
    return prices / rate.reindex(prices.index).to_numpy()


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
    hourly = pd.concat(
        [to_hourly_utc(to_eur(s, currency, rates)) for s, currency in raw],
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
