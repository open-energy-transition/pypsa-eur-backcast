# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds hourly gas prices per hub from daily Trading Hub Europe and MIBGAS prices.

``THE`` is the daily average price of exchange trades at the Trading Hub Europe
virtual trading point (``VTPDailyAveragePrice``), published from gas day
2021-10-01. ``MIBGAS`` is the MIBGAS-ES day-ahead index of the Iberian PVB.
Both are in nominal EUR/MWh. Each gas day runs from 06:00 to 06:00 CET/CEST and
is assigned to the snapshots (naive UTC) it covers. The gas day before the
first snapshot day is not retrieved, so the first hours take the first
available price.
"""

import logging
import re
import xml.etree.ElementTree as ET

import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)

GAS_DAY_START_HOUR = 6
GAS_DAY_TIMEZONE = "Europe/Berlin"


def read_the_prices(fn: str) -> pd.Series:
    """Read the daily VTP average price in EUR/MWh per gas day from a THE PricesEnergyImbalance XML export."""
    prices = {}
    for entry in ET.parse(fn).getroot():
        unit = entry.findtext("Unit")
        assert unit == "EUR/MWh", f"Unexpected THE price unit {unit} in {fn}."
        price = entry.findtext("VTPDailyAveragePrice")
        if price:
            prices[pd.Timestamp(entry.findtext("Gasday"))] = float(price)
    return pd.Series(prices, dtype=float).sort_index()


def read_mibgas_prices(fn: str) -> pd.Series:
    """Read the MIBGAS-ES day-ahead index in EUR/MWh per gas day from an annual MIBGAS data file."""
    sheets = pd.read_excel(fn, sheet_name=None)
    if "MIBGAS Indexes" in sheets:
        df = sheets["MIBGAS Indexes"]
    else:
        # Files up to 2022 list ES and PT in rows of one "Indices" sheet.
        df = sheets["Indices"]
        df = df[df["Area"].astype(str).str.strip() == "ES"]
    names = {c: re.sub(r"\s+", " ", c.split("[")[0]).strip() for c in df.columns}
    column = [
        c for c, name in names.items() if name in ("MIBGAS-ES Index", "MIBGAS Index")
    ]
    assert len(column) == 1, f"No unique MIBGAS-ES index column in {fn}: {column}"
    prices = pd.Series(
        df[column[0]].to_numpy(dtype=float), index=pd.to_datetime(df["Delivery day"])
    )
    return prices.dropna().sort_index()


def gas_days(snapshots: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the gas day of each naive UTC snapshot."""
    local = snapshots.tz_localize("UTC").tz_convert(GAS_DAY_TIMEZONE)
    shifted = local - pd.Timedelta(hours=GAS_DAY_START_HOUR)
    return pd.DatetimeIndex(shifted.tz_localize(None).normalize())


def hourly_gas_price(daily: pd.Series, snapshots: pd.DatetimeIndex) -> pd.Series:
    """Assign daily gas prices to snapshots, filling only the gas day before the first snapshot day."""
    days = gas_days(snapshots)
    first = snapshots[0].normalize()
    missing = days.unique().difference(daily.index)
    assert missing[missing >= first].empty, (
        f"Missing daily gas prices for {missing[missing >= first].strftime('%Y-%m-%d').tolist()}."
    )
    price = daily.reindex(days).bfill()
    assert price.notna().all(), "No daily gas price on the first snapshot day."
    return pd.Series(price.to_numpy(), index=snapshots)


READERS = {"THE": read_the_prices, "MIBGAS": read_mibgas_prices}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_daily_gas_prices")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )

    hourly = {}
    for hub, reader in READERS.items():
        if hub not in snakemake.input.keys():
            continue
        daily = pd.concat([reader(fn) for fn in snakemake.input[hub]])
        daily = daily[~daily.index.duplicated(keep="last")].sort_index()
        hourly[hub] = hourly_gas_price(daily, snapshots)
        logger.info(
            f"{hub} gas price in EUR/MWh: mean {hourly[hub].mean():.1f}, "
            f"min {hourly[hub].min():.1f}, max {hourly[hub].max():.1f}."
        )

    pd.DataFrame(hourly).rename_axis("snapshot").to_csv(snakemake.output.gas_price)
