# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the actual generation per production type of one bidding zone and year from the ENTSO-E Transparency Platform.

The generation is queried through the ENTSO-E API for the calendar year in UTC
and stored in its native market time unit with UTC timestamps. Only the
aggregated generation is kept, the consumption of e.g. pumped storage is
dropped. An API token must be set in the environment variable
``ENTSOE_API_TOKEN``. Areas without published data, e.g. ``GB`` after 2020,
raise an error and must be listed in ``backcast: exclude_generation_areas``.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, utc_year

logger = logging.getLogger(__name__)


def aggregated_generation(generation: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """
    Keep the aggregated generation per production type with UTC timestamps before `end`.

    The ENTSO-E client concatenates the yearly query from shorter periods, so
    plain production type columns can be mixed with ``(type, "Actual
    Aggregated")`` and ``(type, "Actual Consumption")`` columns of periods with
    published consumption, e.g. ``SI`` in October 2020. Plain columns count as
    aggregated generation and are merged with the tuple columns of the same type.
    """
    columns = [
        c if isinstance(c, tuple) else (c, "Actual Aggregated") for c in generation
    ]
    generation = generation.set_axis(pd.MultiIndex.from_tuples(columns), axis=1)
    generation = generation.xs("Actual Aggregated", axis=1, level=1)
    generation = generation.T.groupby(level=0, sort=False).first().T
    generation = generation.tz_convert("UTC").loc[lambda df: df.index < end]
    generation.index.name = "utc_timestamp"
    return generation


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_actual_generation_entsoe", area="DE_LU", year="2023"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    area = snakemake.wildcards.area
    year = int(snakemake.wildcards.year)
    start, end = utc_year(year)

    logger.info(f"Querying actual generation for {area} in {year}...")

    generation = entsoe_client().query_generation(area, start=start, end=end)
    aggregated_generation(generation, end).to_csv(snakemake.output.csv)
