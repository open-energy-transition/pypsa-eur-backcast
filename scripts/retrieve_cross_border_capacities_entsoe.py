# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the day-ahead cross-border capacity data of one bidding zone border, direction and year from the ENTSO-E Transparency Platform.

Three series are queried through the ENTSO-E API for the calendar year in UTC
in monthly chunks: the day-ahead offered capacity of implicit allocations
(``offered_capacity``), the day-ahead net transfer capacity (``ntc``) and the
day-ahead scheduled commercial exchanges (``scheduled_exchange``). Borders
with an area of ``backcast: external_flows: areas`` also get the total
scheduled commercial exchanges (``scheduled_exchange_total``), as most borders
to areas outside the model publish no day-ahead exchanges, and the physical
flows (``physical_flow``), as FI-RU publishes no scheduled exchanges. Capacities
are not published for borders within the flow-based Core region, for which the
build step uses the scheduled exchanges as proxy. Months without published
data are stored as missing values. All series are stored in their native
market time unit with UTC timestamps. An API token must be set in the
environment variable ``ENTSOE_API_TOKEN``.
"""

import logging
from collections.abc import Callable

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, to_utc, utc_year

logger = logging.getLogger(__name__)


def query_monthly(
    query: Callable[[pd.Timestamp, pd.Timestamp], pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    name: str,
) -> pd.Series:
    """Run an ENTSO-E query per month and skip months without published data."""
    bounds = pd.date_range(start, end, freq="MS")
    chunks = []
    for chunk_start, chunk_end in zip(bounds[:-1], bounds[1:]):
        try:
            chunks.append(to_utc(query(chunk_start, chunk_end), chunk_end, name))
        except NoMatchingDataError:
            logger.info(f"No {name} data published from {chunk_start:%Y-%m}.")
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=name)
    return pd.concat([empty, *chunks]).rename(name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_cross_border_capacities_entsoe",
            area_from="FR",
            area_to="ES",
            year="2023",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    area_from = snakemake.wildcards.area_from
    area_to = snakemake.wildcards.area_to
    year = int(snakemake.wildcards.year)
    start, end = utc_year(year)
    client = entsoe_client()

    logger.info(f"Querying cross-border capacities {area_from}->{area_to} in {year}...")

    queries = {
        "offered_capacity": lambda s, e: client.query_offered_capacity(
            area_from, area_to, "A01", start=s, end=e
        ),
        "ntc": lambda s, e: client.query_net_transfer_capacity_dayahead(
            area_from, area_to, start=s, end=e
        ),
        "scheduled_exchange": lambda s, e: client.query_scheduled_exchanges(
            area_from, area_to, start=s, end=e, dayahead=True
        ),
    }
    if {area_from, area_to} & set(snakemake.params.external_areas):
        queries["scheduled_exchange_total"] = lambda s, e: (
            client.query_scheduled_exchanges(
                area_from, area_to, start=s, end=e, dayahead=False
            )
        )
        queries["physical_flow"] = lambda s, e: client.query_crossborder_flows(
            area_from, area_to, start=s, end=e
        )
    data = pd.concat(
        [query_monthly(q, start, end, name) for name, q in queries.items()], axis=1
    )
    data.index.name = "utc_timestamp"
    data.insert(0, "area_to", area_to)
    data.insert(0, "area_from", area_from)

    data.to_csv(snakemake.output.csv)
