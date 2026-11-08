# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build hydroelectric inflow time series per country.

Runoff from the weather cutout is aggregated over each country shape with
[atlite](https://atlite.readthedocs.io/en/latest/), smoothed and normalised so
that the annual inflow matches the annual hydro generation reported by the
U.S. Energy Information Administration (EIA). Optionally, the EIA statistics
are corrected for capacity changes relative to 2019, missing years are
approximated by a linear regression of generation on annual runoff, and all
years can be normalised to a single reference year. Years without EIA data
default to the median year. Inflows below a configurable threshold are clipped
to zero.

| Field | Dimensions | Description |
| --- | --- | --- |
| inflow | countries, time | Inflow to the state of charge in MW, e.g. river inflow to hydro reservoirs |

With ``renewable: hydro: inflow_source: entsoe_balance`` the output is a dataset
that adds the water balance per ENTSO-E bidding zone area: the run-of-river
generation and the reservoir inflow follow the hourly actual generation of the
ENTSO-E carriers ``ror`` and ``hydro`` plus the weekly change of the reported
reservoir filling, and the reservoir energy capacity is the maximum observed
filling. Filling of a country area, e.g. ``IT``, is split onto its zones by
their hydro and run-of-river generation. The annual filling change is spread over the weeks
by their length so that the inflow matches the actual generation over a cyclic
year. Weeks with more drawdown than reservoir generation take the deficit from
the run-of-river generation. Areas without published generation and the areas
in ``entsoe_balance: exclude_areas`` keep the EIA based inflow.

| Field | Dimensions | Description |
| --- | --- | --- |
| ror | time, area | Run-of-river generation in MW |
| reservoir | time, area | Inflow to hydro reservoirs in MW |
| e_max | area | Maximum observed reservoir filling in MWh, NaN without filling data |
| bus_area | bus | ENTSO-E area of each model bus |

![](../img/inflow-ts.png)

![](../img/inflow-box.png)
"""

import logging

import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)
from scripts.lib.entsoe import (
    assert_known_areas,
    read_area_to_bus,
    read_hourly_generation,
)

cc = coco.CountryConverter()


def get_eia_annual_hydro_generation(
    fn: str, countries: list[str], capacities: bool = False
) -> pd.DataFrame:
    # in billion kWh/a = TWh/a
    df = pd.read_csv(fn, skiprows=2, index_col=1, na_values=[" ", "--"]).iloc[1:, 1:]
    df.index = df.index.str.strip()
    df.columns = df.columns.astype(int)

    former_countries = {
        "Former Czechoslovakia": dict(
            countries=["Czechia", "Slovakia"], start=1980, end=1992
        ),
        "Former Serbia and Montenegro": dict(
            countries=["Serbia", "Montenegro", "Kosovo"], start=1992, end=2005
        ),
        "Former Yugoslavia": dict(
            countries=[
                "Slovenia",
                "Croatia",
                "Bosnia and Herzegovina",
                "Serbia",
                "Kosovo",
                "Montenegro",
                "North Macedonia",
            ],
            start=1980,
            end=1991,
        ),
    }

    for k, v in former_countries.items():
        period = [i for i in range(v["start"], v["end"] + 1)]
        ratio = df.loc[v["countries"]].T.dropna().sum()
        ratio /= ratio.sum()
        for country in v["countries"]:
            df.loc[country, period] = df.loc[k, period] * ratio[country]

    baltic_states = ["Latvia", "Estonia", "Lithuania"]
    df.loc[baltic_states] = (
        df.loc[baltic_states].T.fillna(df.loc[baltic_states].mean(axis=1)).T
    )

    df.loc["Germany"] = df.filter(like="Germany", axis=0).sum()
    df = df.loc[~df.index.str.contains("Former")]
    df.drop(["Europe", "Germany, West", "Germany, East"], inplace=True)

    df.index = cc.convert(df.index, to="iso2")
    df.index.name = "countries"

    # convert to MW of MWh/a
    factor = 1e3 if capacities else 1e6
    df = df.T[countries] * factor

    df.ffill(axis=0, inplace=True)

    return df


def correct_eia_stats_by_capacity(
    eia_stats: pd.DataFrame, fn: str, countries: list[str], baseyear: int = 2019
) -> None:
    cap = get_eia_annual_hydro_generation(fn, countries, capacities=True)
    ratio = cap / cap.loc[baseyear]
    eia_stats_corrected = eia_stats / ratio
    to_keep = ["AL", "AT", "CH", "DE", "GB", "NL", "RS", "XK", "RO", "SK"]
    to_correct = eia_stats_corrected.columns.difference(to_keep)
    eia_stats.loc[:, to_correct] = eia_stats_corrected.loc[:, to_correct]


def approximate_missing_eia_stats(
    eia_stats: pd.DataFrame, runoff_fn: str, countries: list[str]
) -> pd.DataFrame:
    runoff = pd.read_csv(runoff_fn, index_col=0, parse_dates=True)[countries]
    runoff = runoff.groupby(runoff.index.year).sum().index

    # fix outliers; exceptional floods in 1977-1979 in ES & PT
    if "ES" in runoff:
        runoff.loc[1978, "ES"] = runoff.loc[1979, "ES"]
    if "PT" in runoff:
        runoff.loc[1978, "PT"] = runoff.loc[1979, "PT"]

    runoff_eia = runoff.loc[eia_stats.index]

    eia_stats_approximated = {}

    for c in countries:
        X = runoff_eia[c]
        Y = eia_stats[c]

        to_predict = runoff.index.difference(eia_stats.index)
        X_pred = runoff.loc[to_predict, c]

        p = Polynomial.fit(X, Y, 1)
        Y_pred = p(X_pred)

        eia_stats_approximated[c] = pd.Series(Y_pred, index=to_predict)

    eia_stats_approximated = pd.DataFrame(eia_stats_approximated)
    return pd.concat([eia_stats, eia_stats_approximated]).sort_index()


def read_filling(files: list[str]) -> pd.Series:
    """
    Read the weekly reservoir filling in MWh of one area from yearly files with overlapping weeks.

    ENTSO-E stamps each value at the start of its week, but the value is the
    level at the end of the week. For NO_1 to NO_5 in 2023, the values
    restamped to the following Monday 00:00 deviate by at most 1-8% from the
    filling that NVE measures on Sunday 24:00 (NVE Magasinstatistikk,
    https://biapi.nve.no/magasinstatistikk), against 13-77% at the week start.
    """
    filling = pd.concat(
        [pd.read_csv(fn, index_col=0, parse_dates=True).iloc[:, 0] for fn in files]
    )
    filling = filling[~filling.index.duplicated(keep="last")].sort_index()
    return filling.set_axis(filling.index + pd.Timedelta("7D"))


def zone_filling(filling: pd.DataFrame, energy: pd.Series) -> pd.DataFrame:
    """Map the filling of each reported area onto the zones it covers, split by their hydro generation `energy`."""
    parent = pd.Series(
        {z: z if z in filling else z.split("_")[0] for z in energy.index}
    )
    parent = parent[parent.isin(filling.columns) & (energy > 0)]
    share = energy[parent.index] / energy[parent.index].groupby(parent).transform("sum")
    return filling[parent].set_axis(parent.index, axis=1).mul(share, axis=1)


def week_edges(time: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the Monday 00:00 edges within `time` plus its start and end."""
    start, end = time[0], time[-1] + pd.Timedelta("1h")
    inner = pd.date_range(start, end, freq="W-MON", inclusive="neither")
    return inner.union(pd.DatetimeIndex([start, end]))


def filling_at(filling: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate the filling of each zone linearly at `times`."""
    x = times.asi8
    return pd.DataFrame(
        {
            z: np.interp(x, s.dropna().index.asi8, s.dropna().values)
            for z, s in filling.items()
        },
        index=times,
    )


def water_balance(
    hydro: pd.DataFrame, ror: pd.DataFrame, filling: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the hourly run-of-river generation and reservoir inflow in MW per zone from the actual generation and the filling change."""
    time = hydro.index
    edges = week_edges(time)
    week = edges[edges.searchsorted(time, side="right") - 1]
    hours = pd.Series(1.0, index=time).groupby(week).sum()
    level = filling_at(filling, edges).reindex(columns=hydro.columns, fill_value=0.0)
    dfill = level.diff().iloc[1:].set_axis(hours.index)
    dfill -= np.outer(hours / hours.sum(), dfill.sum())
    reservoir = hydro.groupby(week).sum() + dfill
    ror_week = ror.groupby(week).sum()
    ror_kept = (ror_week + reservoir.clip(upper=0)).clip(lower=0)
    ror_factor = (ror_kept / ror_week).where(ror_week > 0, 0.0)
    ror_t = ror * ror_factor.reindex(week).set_axis(time)
    reservoir_t = (
        reservoir.clip(lower=0).div(hours, axis=0).reindex(week).set_axis(time)
    )
    return ror_t, reservoir_t


def entsoe_balance(
    generation: pd.DataFrame,
    filling: pd.DataFrame,
    area_to_bus: pd.Series,
    countries: list[str],
    exclude_areas: list[str],
    time: pd.DatetimeIndex,
) -> xr.Dataset:
    """Build the water balance per ENTSO-E area from hourly actual generation and weekly reservoir filling."""
    assert_known_areas(exclude_areas, area_to_bus, "entsoe_balance: exclude_areas")
    reporting_areas = pd.concat([area_to_bus, area_to_bus.str.split("_").str[0]])
    assert_known_areas(
        filling.columns, reporting_areas, "entsoe_balance: filling_first_year"
    )
    carriers = generation.loc[time]
    hydro = carriers.xs("hydro", axis=1, level=1).reindex(
        columns=carriers.columns.levels[0]
    )
    ror = carriers.xs("ror", axis=1, level=1).reindex(columns=hydro.columns)
    published = hydro.notna().any() | ror.notna().any()
    areas = hydro.columns[published & hydro.columns.str[:2].isin(countries)]
    areas = areas.difference(exclude_areas).astype(object)
    hydro = hydro[areas].interpolate(limit_direction="both").fillna(0.0)
    ror = ror[areas].interpolate(limit_direction="both").fillna(0.0)
    zones = zone_filling(filling, hydro.sum() + ror.sum())
    ror_t, reservoir_t = water_balance(hydro, ror, zones)
    return xr.Dataset(
        {
            "ror": ror_t.rename_axis(index="time", columns="area"),
            "reservoir": reservoir_t.rename_axis(index="time", columns="area"),
            "e_max": zones.max().reindex(areas).rename_axis("area"),
            "bus_area": area_to_bus.astype(object)
            .set_axis(area_to_bus.index.astype(object))
            .rename_axis("bus"),
        }
    )


def balance_inflow(
    plants: pd.DataFrame, balance: xr.Dataset
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Distribute the water balance of each area onto its ``ror`` and ``hydro`` plants by ``p_nom``.

    Areas without one of the two carriers give its water to the other.
    Run-of-river generation above ``p_nom`` spills into the reservoirs of the
    area. Returns the inflow per plant in MW and the energy capacity in MWh
    per reservoir plant, NaN without filling data.
    """
    area = plants.bus.map(balance["bus_area"].to_series())
    plants = plants[area.isin(balance.indexes["area"])].assign(area=area)
    ror_area = balance["ror"].to_pandas()
    reservoir_area = balance["reservoir"].to_pandas()
    has = (
        plants.groupby(["area", "carrier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["ror", "hydro"], fill_value=0)
        > 0
    )
    with_ror, with_reservoir = has.index[has.ror], has.index[has.hydro]
    ror_target = ror_area[with_ror].add(
        reservoir_area[with_ror.difference(with_reservoir)], fill_value=0.0
    )
    reservoir_target = reservoir_area[with_reservoir].add(
        ror_area[with_reservoir.difference(with_ror)], fill_value=0.0
    )
    share = plants.p_nom / plants.groupby(["area", "carrier"]).p_nom.transform("sum")
    ror_plants = plants[plants.carrier == "ror"]
    ror_inflow = (
        ror_target[ror_plants.area].set_axis(ror_plants.index, axis=1)
        * share[ror_plants.index]
    )
    spill = (
        (ror_inflow - ror_plants.p_nom).clip(lower=0).T.groupby(ror_plants.area).sum().T
    )
    reservoir_target = reservoir_target.add(
        spill.reindex(columns=with_reservoir, fill_value=0.0), fill_value=0.0
    )
    reservoir_plants = plants[plants.carrier == "hydro"]
    reservoir_inflow = (
        reservoir_target[reservoir_plants.area].set_axis(reservoir_plants.index, axis=1)
        * share[reservoir_plants.index]
    )
    e_nom = (
        balance["e_max"]
        .to_series()[reservoir_plants.area]
        .set_axis(reservoir_plants.index)
        * share[reservoir_plants.index]
    )
    inflow = pd.concat(
        [ror_inflow.clip(upper=ror_plants.p_nom, axis=1), reservoir_inflow], axis=1
    )
    return inflow, e_nom


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_hydro_profile")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params_hydro = snakemake.params.hydro

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    cutout = load_cutout(snakemake.input.cutout)

    years_in_time = pd.DatetimeIndex(time).year.unique()
    cutout_time = pd.DatetimeIndex(cutout.coords["time"].values)

    full_years_available = all(
        pd.Timestamp(f"{year}-01-01") in cutout_time
        and pd.Timestamp(f"{year}-12-31") in cutout_time
        for year in years_in_time
    )

    if full_years_available:
        mask = [pd.Timestamp(t).year in years_in_time for t in cutout_time]
        cutout = cutout.sel(time=cutout_time[mask])
    else:
        cutout = cutout.sel(time=time)

    countries = snakemake.params.countries
    country_shapes = (
        gpd.read_file(snakemake.input.country_shapes)
        .set_index("name")["geometry"]
        .reindex(countries)
    )
    country_shapes.index.name = "countries"

    fn = snakemake.input.eia_hydro_generation
    eia_stats = get_eia_annual_hydro_generation(fn, countries)

    config_hydro = snakemake.config["renewable"]["hydro"]

    if config_hydro.get("eia_correct_by_capacity"):
        fn = snakemake.input.eia_hydro_capacity
        correct_eia_stats_by_capacity(eia_stats, fn, countries)

    if config_hydro.get("eia_approximate_missing"):
        fn = snakemake.input.era5_runoff
        eia_stats = approximate_missing_eia_stats(eia_stats, fn, countries)

    norm_year = config_hydro.get("eia_norm_year")
    missing_years = years_in_time.difference(eia_stats.index)
    if norm_year:
        eia_stats.loc[years_in_time] = eia_stats.loc[norm_year]
    elif missing_years.any():
        for year in missing_years:
            eia_stats.loc[year] = eia_stats.median()

    inflow = cutout.runoff(
        shapes=country_shapes,
        smooth=True,
        lower_threshold_quantile=True,
        normalize_using_yearly=eia_stats,
    )

    if full_years_available:
        inflow = inflow.sel(time=time)

    if "clip_min_inflow" in params_hydro:
        inflow = inflow.where(inflow > params_hydro["clip_min_inflow"], 0)

    if params_hydro["inflow_source"] == "entsoe_balance":
        generation = read_hourly_generation(snakemake.input.entsoe_generation)
        first_year = params_hydro["entsoe_balance"]["filling_first_year"]
        filling = pd.DataFrame(
            {
                area: read_filling(snakemake.input[f"filling_{area}"])
                for area in first_year
                if area[:2] in countries
            }
        )
        balance = entsoe_balance(
            generation,
            filling,
            read_area_to_bus(snakemake.input.area_to_bus),
            countries,
            params_hydro["entsoe_balance"]["exclude_areas"],
            pd.DatetimeIndex(time),
        )
        logger.info(
            "Hydro water balance in TWh per area:\n%s",
            (balance[["ror", "reservoir"]].sum("time").to_pandas() / 1e6).round(2),
        )
        balance.assign(inflow=inflow).to_netcdf(snakemake.output.profile)
    else:
        inflow.rename("inflow").to_netcdf(snakemake.output.profile)
