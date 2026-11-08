# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the weekly filling of water reservoirs and hydro storage plants of one area and year from the ENTSO-E Transparency Platform.

The aggregated filling in MWh is queried through the ENTSO-E API from the
start of the year to the first week of the next year in UTC, so that
consecutive years overlap by one week. Timestamps are the week starts in naive
UTC. An API token must be set in the environment variable ``ENTSOE_API_TOKEN``.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, utc_year

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_reservoir_filling_entsoe", area="NO_2", year="2023"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    area = snakemake.wildcards.area
    year = int(snakemake.wildcards.year)
    start, end = utc_year(year)

    logger.info(f"Querying reservoir filling for {area} in {year}...")

    filling = entsoe_client().query_aggregate_water_reservoirs_and_hydro_storage(
        area, start=start, end=end + pd.Timedelta("7D")
    )
    filling.tz_convert("UTC").tz_localize(None).rename_axis("utc_timestamp").rename(
        "filling_MWh"
    ).to_csv(snakemake.output.csv)
