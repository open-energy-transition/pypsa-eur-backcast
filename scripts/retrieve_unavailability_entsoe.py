# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieves the unavailability of generation and production units of one bidding zone and year from the ENTSO-E Transparency Platform.

Both the unavailability of generation units (document type ``A80``) and of
production units (``A77``) overlapping the calendar year in UTC are queried
through the ENTSO-E API, with all revisions of each outage. Each row is one
step of an outage profile with the available capacity ``avail_qty`` in MW
between ``start`` and ``end`` in UTC. ``production_resource_id`` is the EIC
code of the production unit, ``unit_name`` names the generation unit for
``A80``. Areas without any published unavailability yield an empty table. An
API token must be set in the environment variable ``ENTSOE_API_TOKEN``.
"""

import logging

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.entsoe import entsoe_client, utc_year

logger = logging.getLogger(__name__)

QUERIES = {
    "A80": "query_unavailability_of_generation_units",
    "A77": "query_unavailability_of_production_units",
}
COLUMNS = {
    "mrid": "mrid",
    "revision": "revision",
    "docstatus": "docstatus",
    "businesstype": "businesstype",
    "production_resource_id": "production_resource_id",
    "production_resource_name": "production_resource_name",
    "production_resource_psr_name": "unit_name",
    "plant_type": "plant_type",
    "nominal_power": "nominal_power",
    "avail_qty": "avail_qty",
    "start": "start",
    "end": "end",
}


def outage_steps(outages: pd.DataFrame, doctype: str) -> pd.DataFrame:
    """Keep the outage steps with renamed columns and UTC timestamps."""
    steps = outages.reset_index(drop=True)[list(COLUMNS)].rename(columns=COLUMNS)
    for column in ["start", "end"]:
        steps[column] = pd.to_datetime(steps[column], utc=True)
    steps["avail_qty"] = pd.to_numeric(steps["avail_qty"])
    return steps.assign(doctype=doctype)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_unavailability_entsoe", area="FI", year="2023"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    area = snakemake.wildcards.area
    year = int(snakemake.wildcards.year)
    start, end = utc_year(year)
    client = entsoe_client()

    steps = []
    for doctype, query in QUERIES.items():
        logger.info(f"Querying unavailability ({doctype}) for {area} in {year}...")
        try:
            outages = getattr(client, query)(area, start=start, end=end)
        except NoMatchingDataError:
            logger.warning(f"No unavailability ({doctype}) published for {area}.")
            continue
        steps.append(outage_steps(outages, doctype))
    columns = [*COLUMNS.values(), "doctype"]
    result = pd.concat(steps) if steps else pd.DataFrame(columns=columns)
    result[columns].to_csv(snakemake.output.csv, index=False)
