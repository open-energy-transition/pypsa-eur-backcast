# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the day-ahead electricity prices of one bidding zone and year from the ENTSO-E Transparency Platform.

The prices are queried through the ENTSO-E API for the calendar year in UTC and
stored in their native market time unit with UTC timestamps, together with the
currency they are published in (e.g. PLN, RON, BGN or GBP in 2019). An API token
must be set in the environment variable ``ENTSOE_API_TOKEN``. The query fails if
no data is published for the area and year or if the prices of the year are
given in more than one currency.
"""

import logging
import re

from entsoe import EntsoeRawClient

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, to_utc, utc_year

logger = logging.getLogger(__name__)

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
    xml = EntsoeRawClient.query_day_ahead_prices(client, area, start=start, end=end)
    currencies = set(re.findall(r"<currency_Unit\.name>(\w+)<", xml))
    assert len(currencies) == 1, (
        f"Day-ahead prices of {area} in {year} are given in {sorted(currencies)}, not in one currency. Add {area} to backcast: exclude_price_areas."
    )
    (currency,) = currencies
    prices = to_utc(
        client.query_day_ahead_prices(area, start=start, end=end), end, area
    )

    prices.to_frame().assign(currency=currency).to_csv(snakemake.output.csv)
