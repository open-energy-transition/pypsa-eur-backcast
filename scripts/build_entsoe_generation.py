# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds the hourly and monthly actual generation per ENTSO-E bidding zone area and model carrier.

The ENTSO-E production types are mapped to model carriers with
``PRODUCTION_TYPE_CARRIERS``, unmapped types are dropped. The generation is
resampled to hourly means in UTC and restricted to the snapshots. Per area,
carrier and calendar month, the output holds the generation in MWh summed over
the hours with data and the number of these hours. Months without any data are
omitted. The hourly generation in MW with area and carrier columns is written
as second output. Areas in ``backcast: market_data: generation: exclude_areas`` are not read. The output
is used by the backcast calibration in ``compose_network``
(``backcast: renewables: wind: calibrate`` and ``backcast: dispatch: nuclear: availability``).
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)

PRODUCTION_TYPE_CARRIERS = {
    "Biomass": "biomass",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Oil shale": "lignite",
    "Fossil Peat": "lignite",
    "Fossil Coal-derived gas": "coal",
    "Fossil Hard coal": "coal",
    "Fossil Gas": "gas",
    "Fossil Oil": "oil",
    "Nuclear": "nuclear",
    "Solar": "solar",
    "Wind Onshore": "onwind",
    "Wind Offshore": "offwind",
    "Hydro Run-of-river and pondage": "ror",
    "Hydro Water Reservoir": "hydro",
    "Hydro Pumped Storage": "PHS",
    "Geothermal": "geothermal",
    "Waste": "waste",
}


def hourly_carrier_generation(generation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate production types to model carriers as hourly means in MW with naive UTC timestamps."""
    mapped = generation.loc[:, generation.columns.isin(PRODUCTION_TYPE_CARRIERS)]
    by_carrier = mapped.T.groupby(PRODUCTION_TYPE_CARRIERS).sum(min_count=1).T
    return by_carrier.tz_convert("UTC").tz_localize(None).resample("h").mean()


def hourly_generation(
    hourly: dict[str, pd.DataFrame], snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """Combine the hourly generation per area within the snapshots to one table with area and carrier columns in MW."""
    return pd.concat(
        {area: df.reindex(snapshots) for area, df in hourly.items()},
        axis=1,
        names=["area", "carrier"],
    ).rename_axis("snapshot")


def monthly_generation(hourly: pd.DataFrame) -> pd.DataFrame:
    """Sum hourly generation to monthly energy in MWh per area and carrier with the number of hours with data, omitting months without data."""
    grouped = hourly.groupby(hourly.index.to_period("M").to_timestamp().rename("month"))
    stats = pd.concat({"generation": grouped.sum(), "hours": grouped.count()}, axis=1)
    return (
        stats.stack(["area", "carrier"], future_stack=True)
        .reorder_levels(["area", "carrier", "month"])
        .sort_index()
        .astype({"hours": int})
        .query("hours > 0")
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_entsoe_generation")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    raw = {
        area: pd.concat(
            [
                pd.read_csv(fn, index_col=0, parse_dates=True)
                for fn in snakemake.input[area]
            ]
        )
        for area in snakemake.params.areas
    }
    empty = [area for area, df in raw.items() if df.empty]
    assert not empty, (
        f"Empty actual generation files for {empty}. Add the areas to backcast: "
        "market_data: generation: exclude_areas and delete the files."
    )
    hourly = hourly_generation(
        {area: hourly_carrier_generation(df) for area, df in raw.items()}, snapshots
    )

    hourly.to_csv(snakemake.output.hourly)
    monthly_generation(hourly).to_csv(snakemake.output.csv)
