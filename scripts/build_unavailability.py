# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds the hourly unavailable share of the thermal capacity per bus and carrier from the ENTSO-E unavailability of generation and production units.

Of each outage, the latest revision is kept unless it is cancelled or
withdrawn. Generation unit outages take precedence over production unit
outages of the same production unit. The production units are matched to the
power plants of [build_powerplants][] by their EIC code, nuclear units without
one by name and capacity within the ENTSO-E area. The hourly unavailable
capacity of the matched plants is summed per bus and carrier and divided by
the capacity of all plants of the bus and carrier, including plants without
outage data. The second output holds the capacity, the capacity matched to
production units with outages and their ratio per area and carrier. The
output is used with ``backcast: unavailability: enable`` in ``compose_network``.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config
from scripts.add_electricity import read_powerplants
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.unavailability import (
    hourly_unavailable,
    latest_outages,
    match_production_units,
    production_units,
    unavailable_share,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_unavailability")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    steps = pd.concat(
        [
            pd.read_csv(fn, parse_dates=["start", "end"]).assign(area=area)
            for area in snakemake.params.areas
            for fn in snakemake.input[area]
        ]
    )
    steps = latest_outages(steps)
    unavailable = hourly_unavailable(steps, snapshots)

    ppl = read_powerplants(snakemake.input.powerplants)
    ppl["area"] = ppl.bus.map(area_to_bus)
    matches = match_production_units(production_units(steps), ppl)
    share, coverage = unavailable_share(
        unavailable, matches, ppl, snakemake.params.carriers
    )
    logger.info(
        f"Share of capacity matched to ENTSO-E production units with outages:\n"
        f"{coverage.coverage.unstack().round(2)}"
    )

    share.to_csv(snakemake.output.csv)
    coverage.to_csv(snakemake.output.coverage)
