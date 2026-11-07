# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Builds hourly cross-border capacities between the buses of the bidding zone clustered network.

For each direction of a bidding zone border, the hourly capacity is taken from
the first ENTSO-E source with data: the day-ahead net transfer capacity, the
day-ahead offered capacity of implicit allocations, or, for borders within the
flow-based Core region without published capacities, the monthly maximum of
the day-ahead scheduled commercial exchanges. As the scheduled exchanges are
netted per hour, the monthly maximum is taken over both directions of the
border, so that the capacity of the less used direction is not zero. The net
transfer capacity takes precedence, as the offered capacities of the Italian
internal borders are published with swapped directions. Gaps of the chosen
source up to ``backcast: max_capacity_gap_hours`` are interpolated, longer gaps
raise an error, unless ``backcast: exchange_proxy_fallback`` fills them with
the monthly maximum of the scheduled exchanges, e.g. for the Nordic borders
after the start of flow-based market coupling on 2024-10-29. Borders between two
areas of ``backcast: exchange_proxy_by_direction_areas`` take the monthly maximum
per direction instead. Within a flow-based region without published capacities,
e.g. the Nordic region from 2024-10-29, the maximum over both directions gives
rarely used directions the capacity of the busy one, e.g. NO_1->NO_3 2,918 MW
against 3 MW offered before the go-live, which removes the price differences
within the region.

Some published net transfer capacities are outdated and lower than the
scheduled exchanges, e.g. on the borders within the Core region that still
publish them. Therefore, the capacity is raised to the hourly scheduled
exchange where it is lower, so that the observed exchanges remain feasible.
Borders listed in ``backcast: exchange_proxy_borders`` use the monthly maximum
of the scheduled exchanges regardless of published capacities. All series are
converted to UTC before resampling to hourly means.

The borders are the lines and DC links between buses of different ENTSO-E
areas in ``backcast: area_to_bus``. If several bus pairs share one area
border, its capacity is split in proportion to their physical transmission
capacity. Copperplate connections and connections within one area are not
considered. Borders listed in ``backcast: fixed_border_capacities`` get a
constant capacity in both directions instead, either in MW or the physical
capacity of the network. The build fails if any other border lacks data, e.g.
if it is missing in ``backcast: borders``, which lists the borders to
retrieve.

The first output holds the capacities in MW with columns ``<bus0>-><bus1>``,
the second output the source and number of interpolated hours per direction,
the third output the hourly scheduled commercial exchanges in MW per retrieved
area direction ``<area0>-><area1>``, NaN where unpublished.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config
from scripts.lib.cross_border import area_border, border_name, inter_zone_branches
from scripts.lib.entsoe import read_area_to_bus
from scripts.lib.timeseries import fill_short_gaps, gap_lengths

logger = logging.getLogger(__name__)

SOURCES = ["ntc", "offered_capacity", "scheduled_exchange"]


def border_shares(branches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate branches with positive capacity per bus pair with their share of the physical capacity of the area border."""
    pairs = (
        branches.query("capacity > 0")
        .groupby(["bus0", "bus1", "area0", "area1"])["capacity"]
        .sum()
        .reset_index()
    )
    pairs["border"] = [area_border(a0, a1) for a0, a1 in zip(pairs.area0, pairs.area1)]
    pairs["share"] = pairs.capacity / pairs.groupby("border").capacity.transform("sum")
    return pairs


def area_capacities(
    data: pd.DataFrame,
    borders: list[tuple[str, str]],
    snapshots: pd.DatetimeIndex,
    max_gap: int,
    exchange_proxy_borders: list[str],
    exchange_proxy_fallback: bool = False,
    exchange_proxy_by_direction_areas: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return hourly capacities for both directions of each area border with gaps up to `max_gap` hours interpolated, longer gaps filled with the exchange proxy if `exchange_proxy_fallback`, raised to the scheduled exchange, and the source and number of interpolated hours per direction. The exchange proxy is the monthly maximum over both directions, or per direction for borders between two areas of `exchange_proxy_by_direction_areas`."""
    directions = sorted({*borders, *(b[::-1] for b in borders)})
    names = [border_name(*d) for d in directions]
    missing = [name for d, name in zip(directions, names) if d not in data.index]
    assert not missing, (
        f"No cross-border capacity data for {missing}, check `backcast: borders`."
    )
    hourly = {
        name: data.loc[d, SOURCES].resample("h").mean().reindex(snapshots)
        for d, name in zip(directions, names)
    }
    exchange = pd.DataFrame(
        {name: h.scheduled_exchange for name, h in hourly.items()}, index=snapshots
    )
    reverse = exchange[[border_name(b, a) for a, b in directions]].set_axis(
        names, axis=1
    )
    by_direction = [
        name
        for d, name in zip(directions, names)
        if set(d) <= set(exchange_proxy_by_direction_areas)
    ]
    reverse[by_direction] = exchange[by_direction]
    proxy = (
        np.fmax(exchange, reverse).groupby(snapshots.to_period("M")).transform("max")
    )
    proxy.loc[:, exchange.isna().all()] = np.nan
    raw, source = {}, {}
    for d, name in zip(directions, names):
        candidates = hourly[name].assign(scheduled_exchange=proxy[name])
        if area_border(*d) in exchange_proxy_borders:
            candidates = candidates[["scheduled_exchange"]]
        available = candidates.columns[candidates.notna().any()]
        assert not available.empty, f"No cross-border capacity data for {name}."
        raw[name], source[name] = candidates[available[0]], available[0]
        if exchange_proxy_fallback and source[name] != "scheduled_exchange":
            long_gaps = gap_lengths(raw[name]) > max_gap
            if long_gaps.any():
                raw[name] = raw[name].where(~long_gaps, proxy[name])
                source[name] += "+scheduled_exchange"
    raw = pd.DataFrame(raw, index=snapshots)
    capacities = np.fmax(fill_short_gaps(raw, max_gap, max_gap), exchange)
    info = pd.DataFrame({"source": source, "interpolated_hours": raw.isna().sum()})
    return capacities, info


def build_capacities(
    data: pd.DataFrame,
    pairs: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    fixed: dict[str, float | str],
    max_gap: int,
    exchange_proxy_borders: list[str],
    exchange_proxy_fallback: bool = False,
    exchange_proxy_by_direction_areas: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map directed area capacities to bus pairs, use `fixed` capacities for listed area borders and record the source and interpolated hours per direction."""
    measured = pairs[~pairs.border.isin(fixed)]
    borders = list(zip(measured.area0, measured.area1))
    area, info = area_capacities(
        data,
        borders,
        snapshots,
        max_gap,
        exchange_proxy_borders,
        exchange_proxy_fallback,
        exchange_proxy_by_direction_areas,
    )
    capacities = {}
    sources = {}
    for pair in pairs.itertuples():
        fixed_capacity = fixed.get(pair.border)
        for bus_from, bus_to, area_from, area_to in [
            (pair.bus0, pair.bus1, pair.area0, pair.area1),
            (pair.bus1, pair.bus0, pair.area1, pair.area0),
        ]:
            name = border_name(bus_from, bus_to)
            area_name = border_name(area_from, area_to)
            if fixed_capacity is None:
                capacities[name] = area[area_name] * pair.share
                source = info.loc[area_name].to_dict()
            else:
                value = (
                    pair.capacity
                    if fixed_capacity == "physical"
                    else fixed_capacity * pair.share
                )
                capacities[name] = pd.Series(float(value), snapshots)
                source = {"source": "fixed", "interpolated_hours": 0}
            sources[name] = {"area_from": area_from, "area_to": area_to, **source}
    return pd.DataFrame(capacities, index=snapshots), pd.DataFrame(
        sources
    ).T.rename_axis("border")


def scheduled_exchanges(
    data: pd.DataFrame, snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """Return the hourly scheduled commercial exchanges per retrieved area direction in MW, NaN where unpublished."""
    directions = data.index.droplevel("utc_timestamp").unique()
    return pd.DataFrame(
        {
            border_name(*d): data.loc[d, "scheduled_exchange"]
            .resample("h")
            .mean()
            .reindex(snapshots)
            for d in directions
        },
        index=snapshots,
    )


def read_retrieved(fns: list[str]) -> pd.DataFrame:
    """Read retrieved cross-border data indexed by area pair and UTC timestamp."""
    data = pd.concat(
        [pd.read_csv(fn, dtype=dict.fromkeys(SOURCES, float)) for fn in fns]
    )
    data["utc_timestamp"] = pd.to_datetime(data.utc_timestamp, utc=True).dt.tz_localize(
        None
    )
    return data.set_index(["area_from", "area_to", "utc_timestamp"]).sort_index()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_cross_border_capacities")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    n = pypsa.Network(snakemake.input.network)

    pairs = border_shares(inter_zone_branches(n, area_to_bus))
    data = read_retrieved(snakemake.input.capacities)
    capacities, sources = build_capacities(
        data,
        pairs,
        snapshots,
        snakemake.params.fixed_border_capacities,
        snakemake.params.max_capacity_gap_hours,
        snakemake.params.exchange_proxy_borders,
        snakemake.params.exchange_proxy_fallback,
        snakemake.params.exchange_proxy_by_direction_areas,
    )

    logger.info(f"Cross-border capacity sources:\n{sources.to_string()}")

    capacities.to_csv(snakemake.output.capacities)
    sources.to_csv(snakemake.output.sources)
    scheduled_exchanges(data, snapshots).to_csv(snakemake.output.exchanges)
