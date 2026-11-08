# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for the transmission capacities between ENTSO-E bidding zone areas.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts.lib.entsoe import assert_known_areas

logger = logging.getLogger(__name__)


def border_name(bus0: str, bus1: str) -> str:
    """Return the column name of the direction from `bus0` to `bus1`."""
    return f"{bus0}->{bus1}"


def area_border(area0: str, area1: str) -> str:
    """Return the undirected name of the border between two areas."""
    return "-".join(sorted([area0, area1]))


def inter_zone_branches(n: pypsa.Network, area_to_bus: pd.Series) -> pd.DataFrame:
    """Return lines and DC links between buses of different areas with their nominal capacity, length and underwater fraction, ordered by bus name."""
    lines = n.lines.assign(capacity=n.lines.s_nom, underwater_fraction=0.0)
    links = n.links.query("carrier == 'DC'").assign(capacity=lambda df: df.p_nom)
    columns = ["bus0", "bus1", "capacity", "length", "underwater_fraction"]
    branches = pd.concat(
        {"Line": lines[columns], "Link": links[columns]}, names=["component", "name"]
    )
    swap = branches.bus0 > branches.bus1
    branches.loc[swap, ["bus0", "bus1"]] = branches.loc[swap, ["bus1", "bus0"]].values
    branches["area0"] = branches.bus0.map(area_to_bus)
    branches["area1"] = branches.bus1.map(area_to_bus)
    unmapped = branches[["area0", "area1"]].isna().any(axis=1)
    assert not unmapped.any(), (
        f"Branches with buses without ENTSO-E area: {branches[unmapped].index.tolist()}"
    )
    return branches.query("area0 != area1")


def raise_to_physical(
    n: pypsa.Network,
    capacities: pd.DataFrame,
    area_to_bus: pd.Series,
    sources: pd.DataFrame,
    physical_factor: float,
    dc_physical_factor: float,
    proxy_sources: list[str],
    areas: list[str],
) -> pd.DataFrame:
    """
    Raise the capacities of directions with a source in `proxy_sources` between two of the `areas` to a share of the physical capacity of their bus pair.

    The lower bound is the sum of `physical_factor` times ``s_nom`` of the
    lines and `dc_physical_factor` times ``p_nom`` of the DC links between the
    two buses, whose nominal capacities are kept by
    ``backcast: fixed_border_capacities: physical``.
    """
    assert_known_areas(areas, area_to_bus, "backcast: hybrid_capacities: areas")
    branches = inter_zone_branches(n, area_to_bus)
    factors = {"Line": physical_factor, "Link": dc_physical_factor}
    factor = branches.index.get_level_values("component").map(factors)
    physical = (
        (branches.capacity * factor).groupby([branches.bus0, branches.bus1]).sum()
    )
    selected = sources.source.isin(proxy_sources) & (
        sources.area_from.isin(areas) & sources.area_to.isin(areas)
    )
    names = sources.index[selected].intersection(capacities.columns)
    floor = pd.Series(
        [physical[tuple(sorted(name.split("->")))] for name in names], index=names
    )
    raised = capacities.copy()
    raised[names] = capacities[names].clip(lower=floor, axis=1)
    logger.info(
        f"Raised {len(names)} cross-border directions to {physical_factor} x line and "
        f"{dc_physical_factor} x DC link capacity, mean capacity {capacities[names].mean().mean():.0f} -> "
        f"{raised[names].mean().mean():.0f} MW."
    )
    return raised


def apply_cross_border_capacities(
    n: pypsa.Network,
    capacities: pd.DataFrame,
    area_to_bus: pd.Series,
) -> None:
    """
    Replace lines and DC links between bidding zones by one bidirectional link per border.

    Each link gets `p_nom` as the maximum capacity over both directions and
    hourly `p_max_pu` and `p_min_pu` from the capacities of the forward and
    reverse direction. Copperplate connections and connections within one
    bidding zone are kept.

    Parameters
    ----------
    n : pypsa.Network
        Network with hourly snapshots.
    capacities : pd.DataFrame
        Hourly capacities in MW with columns ``<bus0>-><bus1>`` for both directions of each border.
    area_to_bus : pd.Series
        ENTSO-E bidding zone area per bus.
    """
    branches = inter_zone_branches(n, area_to_bus)
    borders = (
        branches.query("capacity > 0")
        .groupby(["bus0", "bus1"])
        .agg(
            length=("length", "mean"),
            underwater_fraction=("underwater_fraction", "mean"),
        )
    )
    forward = [border_name(b0, b1) for b0, b1 in borders.index]
    reverse = [border_name(b1, b0) for b0, b1 in borders.index]
    missing = pd.Index(forward + reverse).difference(capacities.columns)
    assert missing.empty, f"No cross-border capacities for {missing.tolist()}."
    capacities = capacities.reindex(n.snapshots)[forward + reverse]
    assert capacities.notna().all().all(), "Cross-border capacities miss snapshots."

    p_nom = pd.Series(
        np.maximum(capacities[forward].max().values, capacities[reverse].max().values),
        index=forward,
    )
    for c, names in branches.index.to_frame().groupby(level="component").name:
        n.remove(c, names)
    logger.info(
        f"Replacing {len(branches)} lines and links by {len(borders)} cross-border links."
    )
    n.add(
        "Link",
        forward,
        bus0=borders.index.get_level_values("bus0"),
        bus1=borders.index.get_level_values("bus1"),
        carrier="DC",
        p_nom=p_nom,
        p_max_pu=capacities[forward].div(p_nom).fillna(0.0),
        p_min_pu=-capacities[reverse].set_axis(forward, axis=1).div(p_nom).fillna(0.0),
        length=borders.length.values,
        underwater_fraction=borders.underwater_fraction.values,
    )


def external_net_imports(
    data: pd.DataFrame,
    borders: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    source: str,
) -> pd.DataFrame:
    """
    Return the hourly net import in MW from the external area into the modelled area of each border.

    The columns are named ``<external_area>-><area>``. Borders with ``flow``
    ``scheduled`` use the scheduled exchanges `source`, borders with ``flow``
    ``physical`` the physical flows. A direction without a value in an hour
    counts as zero, hours in which neither direction has a value are logged
    as a warning per border, and a border without any value raises.
    """
    assert set(borders.flow) <= {"scheduled", "physical"}, (
        f"Unknown `flow` in external borders: {sorted(set(borders.flow))}"
    )
    series = borders.flow.map({"scheduled": source, "physical": "physical_flow"})
    absent = sorted(set(series) - set(data.columns))
    assert not absent, (
        f"Retrieved cross-border data lacks {absent}, retrieve it again with the "
        "area in `backcast: external_flows: areas`."
    )
    hourly = {
        column: data[column]
        .astype(float)
        .unstack(["area_from", "area_to"])
        .resample("h")
        .mean()
        .reindex(snapshots)
        for column in set(series)
    }
    nan = pd.Series(np.nan, snapshots)
    net = {}
    missing = {}
    for column, area, external in zip(series, borders.area, borders.external_area):
        name = border_name(external, area)
        imports = hourly[column].get((external, area), nan)
        exports = hourly[column].get((area, external), nan)
        missing[name] = int((imports.isna() & exports.isna()).sum())
        net[name] = imports.fillna(0.0) - exports.fillna(0.0)
    missing = pd.Series(missing, dtype=int)
    empty = missing.index[missing == len(snapshots)].tolist()
    assert not empty, (
        f"No published exchanges in any snapshot for external borders {empty}."
    )
    if missing.any():
        logger.warning(
            f"Hours without published exchanges per external border, counted as zero:\n{missing[missing > 0].to_string()}"
        )
    return pd.DataFrame(net, index=snapshots)


def add_external_flows(
    n: pypsa.Network,
    net_imports: pd.DataFrame,
    borders: pd.DataFrame,
    area_to_bus: pd.Series,
) -> None:
    """
    Add fixed hourly flows to areas outside the model at the bus of the modelled area with the largest load.

    Net imports become a generator with ``p_min_pu`` equal to ``p_max_pu``
    (carrier ``external import``), net exports a load (carrier ``external
    export``), each named ``<external_area>-><area> import|export``.
    """
    modelled = area_to_bus[area_to_bus.index.isin(n.buses.index)]
    overlap = sorted(set(borders.external_area) & set(modelled))
    assert not overlap, (
        f"External areas are modelled as buses, remove them from `backcast: external_flows: areas`: {overlap}"
    )
    load = n.loads_t.p_set.sum().groupby(n.loads.bus).sum().reindex(modelled.index)
    main_bus = load.fillna(0.0).groupby(modelled).idxmax()
    bus = borders.area.map(main_bus).set_axis(net_imports.columns)
    assert bus.notna().all(), (
        f"Border areas without bus: {bus[bus.isna()].index.tolist()}"
    )
    imports = net_imports.clip(lower=0.0).loc[:, lambda df: df.max() > 0]
    exports = (-net_imports).clip(lower=0.0).loc[:, lambda df: df.max() > 0]
    p_nom = imports.max()
    p_pu = imports.div(p_nom).add_suffix(" import")
    n.add(
        "Generator",
        p_pu.columns,
        bus=bus[imports.columns].values,
        carrier="external import",
        p_nom=p_nom.values,
        p_max_pu=p_pu,
        p_min_pu=p_pu,
        marginal_cost=0.0,
    )
    n.add(
        "Load",
        exports.columns + " export",
        bus=bus[exports.columns].values,
        carrier="external export",
        p_set=exports.add_suffix(" export"),
    )
    net = net_imports.mul(n.snapshot_weightings.generators, axis=0).sum().div(1e6)
    logger.info(
        f"Fixed net imports from external areas in TWh:\n{net.round(2).to_string()}"
    )
