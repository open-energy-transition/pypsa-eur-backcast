# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds the hourly heat-led minimum output of CHP plants per country and fuel group.

The annual floor energy of a country and fuel group is the minimum of the
back-pressure ratio ``c_b`` of the matching CHP technology in the cost data times
the main-activity CHP heat output and the main-activity CHP electricity output in
Eurostat ``nrg_bal_peh`` for the snapshot year or ``year``. Autoproducer CHP serves on-site
demand and is not counted. Oil counts the marketed fuels (fuel oil, gas oil, LPG,
naphtha, kerosene and gasoline) but not refinery gas and other oil products: these
refinery by-products fire refinery-sited plants that raise process steam, which
does not follow heating degree days. Excluding them lowers the oil floor in every
year where Eurostat reports them as main-activity CHP. ``c_b`` times the heat is the least electricity a
back-pressure or extraction unit generates for that heat. The floor follows a
daily heat demand profile: heating degree days below ``heat_threshold`` from the
cutout temperature, weighted by the urban population of each country, plus a flat
``hot_water_share`` of the annual heat. The output holds the floor in MW with
country and fuel columns. It is used by ``compose_network`` with
``backcast: chp_floor``.
"""

import logging

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)

FUEL_SIEC = {
    "gas": ["G3000"],
    "coal": ["C0110", "C0121", "C0129", "C0210"],
    "lignite": ["C0220"],
    "oil": [
        "O4630",
        "O4640",
        "O4651",
        "O4652XR5210B",
        "O4653",
        "O4661XR5230B",
        "O4669",
        "O4671XR5220B",
        "O4680",
    ],
    "biomass": ["R5110-5150_W6000RI", "R5300", "R5210B", "R5220B", "R5230B", "R5290"],
    "waste": ["W6100_6220", "W6210"],
}
FUEL_TECHNOLOGY = {
    "gas": "central gas CHP",
    "oil": "central gas CHP",
    "coal": "central coal CHP",
    "lignite": "central coal CHP",
    "biomass": "central solid biomass CHP",
    "waste": "waste CHP",
}
EUROSTAT_COUNTRIES = {"EL": "GR", "UK": "GB"}


def read_chp_production(path: str, year: int, fuels: list[str]) -> pd.DataFrame:
    """Read main-activity CHP electricity and heat output in MWh per country and fuel group from Eurostat ``nrg_bal_peh``."""
    raw = pd.read_csv(path, sep="\t", na_values=":")
    keys = raw.iloc[:, 0].str.split(",", expand=True)
    keys.columns = ["freq", "nrg_bal", "siec", "unit", "geo"]
    columns = {c.strip(): c for c in raw.columns}
    assert str(year) in columns, f"Eurostat nrg_bal_peh has no data for {year}."
    values = raw[columns[str(year)]]
    values = pd.to_numeric(values.astype(str).str.extract(r"([-\d.]+)")[0])
    siec_fuel = {s: f for f in fuels for s in FUEL_SIEC[f]}
    df = keys.assign(value=values, fuel=keys.siec.map(siec_fuel)).query(
        "unit == 'GWH' and nrg_bal in ['GEP_MAPCHP', 'GHP_MAPCHP'] and fuel == fuel"
    )
    df["country"] = df.geo.replace(EUROSTAT_COUNTRIES)
    table = df.pivot_table(
        index=["country", "fuel"], columns="nrg_bal", values="value", aggfunc="sum"
    )
    return table.rename(
        columns={"GEP_MAPCHP": "electricity", "GHP_MAPCHP": "heat"}
    ).mul(1e3)


def eurostat_year(snapshots: pd.DatetimeIndex, year: int | None) -> int:
    """Return the Eurostat reference year, ``year`` if set, else the single snapshot year."""
    years = snapshots.year.unique()
    assert len(years) == 1, (
        f"backcast.chp_floor requires snapshots in one year, got {list(years)}."
    )
    return int(years[0]) if year is None else year


def floor_energy(production: pd.DataFrame, c_b: pd.Series) -> pd.Series:
    """Return the annual floor energy per country and fuel as the minimum of ``c_b`` times the CHP heat and the CHP electricity."""
    ratio = production.index.get_level_values("fuel").map(c_b).to_numpy()
    return np.minimum(ratio * production["heat"], production["electricity"]).dropna()


def heat_profile(
    temperature: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    threshold: float,
    hot_water_share: float,
) -> pd.DataFrame:
    """Return the hourly heat demand share per country, constant within each day and summing to one over the snapshots."""
    degree_days = (threshold - temperature.resample("D").mean()).clip(lower=0)
    daily = degree_days + hot_water_share / (1 - hot_water_share) * degree_days.mean()
    hourly = daily.reindex(snapshots.floor("D")).set_axis(snapshots)
    assert hourly.notna().all().all(), (
        "Cutout temperature does not cover the snapshots."
    )
    return hourly / hourly.sum()


def country_temperature(
    cutout: atlite.Cutout, population: xr.DataArray, shapes: gpd.GeoSeries
) -> pd.DataFrame:
    """Return the hourly mean temperature in degree Celsius per country, weighted by population."""
    weights = cutout.indicatormatrix(shapes).multiply(population.values.ravel())
    weights = weights.multiply(1 / weights.sum(axis=1))
    temperature = cutout.data.temperature.stack(spatial=["y", "x"]) - 273.15
    values = weights @ temperature.values.T
    return pd.DataFrame(
        values.T, index=pd.DatetimeIndex(temperature.time.values), columns=shapes.index
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_chp_floor", run="backcast-2023")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.chp_floor
    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    production = read_chp_production(
        snakemake.input.eurostat,
        eurostat_year(snapshots, params["year"]),
        params["fuels"],
    )
    c_b = pd.read_csv(snakemake.input.tech_costs, index_col=0)["c_b"]
    energy = floor_energy(
        production,
        c_b.reindex(FUEL_TECHNOLOGY.values()).set_axis(FUEL_TECHNOLOGY.keys()),
    )

    shapes = gpd.read_file(snakemake.input.country_shapes).set_index("name").geometry
    shapes = shapes[shapes.index.isin(energy.index.get_level_values("country"))]
    cutout = atlite.Cutout(snakemake.input.cutout)
    population = xr.open_dataarray(snakemake.input.pop_layout)
    temperature = country_temperature(cutout, population, shapes)
    profile = heat_profile(
        temperature, snapshots, params["heat_threshold"], params["hot_water_share"]
    )

    energy = energy[energy.index.get_level_values("country").isin(profile.columns)]
    floor = pd.DataFrame(
        {(c, f): e * profile[c] for (c, f), e in energy.items()}, index=snapshots
    )
    floor.columns.names = ["country", "fuel"]
    logger.info(
        "CHP floor energy in TWh: "
        + energy.div(1e6).round(1).loc[lambda s: s >= 1].to_string().replace("\n", ", ")
    )
    floor.to_csv(snakemake.output[0])
