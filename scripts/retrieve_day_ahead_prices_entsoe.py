# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the day-ahead electricity prices of one bidding zone and year from the ENTSO-E Transparency Platform.

The prices are queried through the ENTSO-E API for the calendar year in UTC and
stored in their native market time unit with UTC timestamps, together with the
currency of the daily document they are published in (e.g. PLN, RON, BGN or GBP
in 2019; RO switches from RON to EUR on delivery day 2021-06-18, BG from BGN to
EUR on 2022-01-22). The currencies are read from the raw documents queried per
month, as the API returns at most 100 documents per query. An API token must be
set in the environment variable ``ENTSOE_API_TOKEN``. The query fails if no data
is published for the area and year or if a document of the same start is given
in two currencies.
"""

import logging
import re

import pandas as pd
from entsoe import EntsoePandasClient, EntsoeRawClient

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, to_utc, utc_year

logger = logging.getLogger(__name__)

DOCUMENT_CURRENCY = re.compile(
    r"<currency_Unit\.name>(\w+)</currency_Unit\.name>.*?<start>([^<]+)</start>",
    flags=re.DOTALL,
)


def parse_document_currencies(xml: str) -> pd.Series:
    """Return the currency of each price document in `xml`, indexed by the UTC start of its period."""
    currency, start = zip(*DOCUMENT_CURRENCY.findall(xml))
    return pd.Series(currency, pd.DatetimeIndex(start, name="start"), name="currency")


def query_document_currencies(
    client: EntsoePandasClient, area: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Query the currency of each daily price document between `start` and `end` month by month."""
    bounds = pd.date_range(start, end, freq="MS")
    documents = pd.concat(
        parse_document_currencies(
            EntsoeRawClient.query_day_ahead_prices(client, area, start=s, end=e)
        )
        for s, e in zip(bounds[:-1], bounds[1:])
    )
    documents = documents.reset_index().drop_duplicates().set_index("start")
    assert documents.index.is_unique, (
        f"Day-ahead price documents of {area} with the same start are given in two currencies."
    )
    return documents["currency"].sort_index()


def currency_per_timestamp(prices: pd.Series, currencies: pd.Series) -> pd.Series:
    """Assign each price the currency of the last document starting at or before its timestamp."""
    result = currencies.reindex(prices.index, method="ffill")
    assert result.notna().all(), f"Prices of {prices.name} before their first document."
    return result


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_day_ahead_prices_entsoe", area="DE_LU", year="2023"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    area = snakemake.wildcards.area
    year = int(snakemake.wildcards.year)
    start, end = utc_year(year)

    logger.info(f"Querying day-ahead prices for {area} in {year}...")

    client = entsoe_client()
    currencies = query_document_currencies(client, area, start, end)
    prices = to_utc(
        client.query_day_ahead_prices(area, start=start, end=end), end, area
    )
    currency = currency_per_timestamp(prices, currencies)
    logger.info(f"Currencies of {area} in {year}: {currency.value_counts().to_dict()}")

    prices.to_frame().assign(currency=currency).to_csv(snakemake.output.csv)
