# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Hourly unavailable capacity of power plants from the ENTSO-E unavailability of generation and production units.
"""

import ast
import logging
import re
import unicodedata

import numpy as np
import pandas as pd
import pypsa

from scripts.lib.entsoe import assert_known_areas

logger = logging.getLogger(__name__)

# ENTSO-E production types matched by name to the powerplantmatching fuel types
# (lower case, as in the PyPSA column names) of plants without an EIC code.
PLANT_TYPE_FUELTYPES = {
    "Fossil Gas": ["natural gas", "CCGT", "OCGT"],
    "Fossil Hard coal": ["coal"],
    "Fossil Coal-derived gas": ["coal"],
    "Fossil Brown coal/Lignite": ["lignite"],
    "Fossil Oil shale": ["lignite"],
    "Fossil Peat": ["lignite"],
    "Fossil Oil": ["oil"],
    "Nuclear": ["nuclear"],
}
INACTIVE_STATUS = ["Cancelled", "Withdrawn"]


def latest_outages(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the latest revision of each outage and drop cancelled and withdrawn outages.

    Outages are identified by their document type and ``mrid``. Generation unit
    outages (``A80``) take precedence over production unit outages (``A77``):
    the latter are only kept for production units without any generation unit
    outage, since both document types may report the same event.
    """
    key = [steps.doctype, steps.mrid]
    steps = steps[steps.revision == steps.groupby(key).revision.transform("max")]
    steps = steps[~steps.docstatus.isin(INACTIVE_STATUS)]
    by_unit = steps.query("doctype == 'A80'").production_resource_id.unique()
    return steps[(steps.doctype == "A80") | ~steps.production_resource_id.isin(by_unit)]


def hourly_mean(level: pd.Series, hours: pd.DatetimeIndex) -> pd.Series:
    """Hourly time-weighted mean of a step function holding each value from its timestamp until the next."""
    grid = pd.date_range(hours[0], hours[-1] + pd.Timedelta(hours=1), freq="h")
    level = level.reindex(level.index.union(grid)).ffill().fillna(0.0)
    level = level[(level.index >= grid[0]) & (level.index <= grid[-1])]
    seconds = (level.index[1:] - level.index[:-1]).total_seconds()
    weighted = pd.Series(level.values[:-1] * seconds, index=level.index[:-1])
    hourly = weighted.groupby(weighted.index.floor("h")).sum() / 3600
    return hourly.reindex(hours, fill_value=0.0)


def unit_profile(steps: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.Series:
    """Hourly mean unavailable capacity of one unit, the maximum over its overlapping outage steps."""
    times = np.unique(np.concatenate([steps.start.values, steps.end.values]))
    active = (steps.start.values[:, None] <= times) & (
        times < steps.end.values[:, None]
    )
    level = np.where(active, steps.unavailable.values[:, None], 0.0).max(axis=0)
    return hourly_mean(pd.Series(level, index=pd.DatetimeIndex(times)), hours)


def hourly_unavailable(steps: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Hourly mean unavailable capacity in MW per production unit.

    The unavailable capacity of an outage step is its nominal power minus the
    available capacity. Overlapping outages of the same generation unit, e.g.
    a forced outage during planned maintenance, are not added up, the larger
    one counts. The generation units of a production unit are summed.

    Parameters
    ----------
    steps : pd.DataFrame
        Outage steps of `latest_outages` with UTC timestamps.
    hours : pd.DatetimeIndex
        Hourly naive UTC timestamps.

    Returns
    -------
    pd.DataFrame
        Unavailable capacity indexed by `hours` with production unit EIC codes as columns.
    """
    steps = steps.assign(
        unit=steps.unit_name.fillna(steps.production_resource_id),
        unavailable=(steps.nominal_power - steps.avail_qty).clip(lower=0),
        start=steps.start.dt.tz_convert("UTC").dt.tz_localize(None),
        end=steps.end.dt.tz_convert("UTC").dt.tz_localize(None),
    )
    period = (steps.end > hours[0]) & (steps.start < hours[-1] + pd.Timedelta(hours=1))
    steps = steps[period & (steps.unavailable > 0)]
    if steps.empty:
        return pd.DataFrame(index=hours, columns=pd.Index([], dtype=str), dtype=float)
    units = {
        key: unit_profile(group, hours)
        for key, group in steps.groupby(["production_resource_id", "unit"])
    }
    units = pd.DataFrame(units, index=hours)
    return units.T.groupby(level=0).sum().T


def entsoe_codes(ppl: pd.DataFrame) -> pd.Series:
    """ENTSO-E EIC codes of the power plants from their powerplantmatching project IDs, one row per code."""

    def codes(project_ids: str) -> list[str]:
        if not isinstance(project_ids, str):
            return []
        return sorted(ast.literal_eval(project_ids).get("ENTSOE", set()))

    return ppl.projectid.map(codes).explode().dropna()


def name_tokens(name: str) -> set[str]:
    """Lower case words and numbers of a plant name without accents."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore")
    return set(re.findall(r"[a-z]+|\d+", ascii_name.decode().lower()))


def name_score(names: list[str], candidate: str) -> float:
    """Share of the words of the shorter name found in the other, the best over `names`."""
    other = name_tokens(candidate)
    scores = [
        len(tokens & other) / min(len(tokens), len(other))
        for tokens in map(name_tokens, names)
        if tokens and other
    ]
    return max(scores, default=0.0)


def match_production_units(
    units: pd.DataFrame,
    ppl: pd.DataFrame,
    min_name_score: float = 0.5,
    capacity_tolerance: float = 0.25,
) -> pd.Series:
    """
    Match ENTSO-E production units to power plants.

    Production units are matched by their EIC code to the ENTSO-E project IDs
    of powerplantmatching. Unmatched production units of the production types
    in ``PLANT_TYPE_FUELTYPES`` are matched to the plant of the same area and
    fuel type without EIC match with the best name score of at least
    `min_name_score` and a capacity within `capacity_tolerance` of the
    production unit, with the closest capacity breaking ties.

    Parameters
    ----------
    units : pd.DataFrame
        Production units indexed by EIC code with ``area``, ``plant_type``,
        ``capacity`` in MW and ``names``, a list of the production and
        generation unit names.
    ppl : pd.DataFrame
        Power plants of `read_powerplants` with ``area``.

    Returns
    -------
    pd.Series
        Power plant index per production unit EIC code, one row per match.
    """
    codes = entsoe_codes(ppl)
    by_code = pd.Series(codes.index, index=codes.values)
    by_code = by_code[by_code.index.isin(units.index)]
    candidates = ppl.drop(by_code.unique())
    matches = [by_code]
    for code, unit in units.drop(by_code.index.unique()).iterrows():
        fueltypes = PLANT_TYPE_FUELTYPES.get(unit.plant_type)
        if not fueltypes:
            continue
        pool = candidates.query("area == @unit.area and carrier in @fueltypes")
        pool = pool[
            (pool.p_nom - unit.capacity).abs() <= capacity_tolerance * unit.capacity
        ]
        pool = pool.assign(
            score=pool.name.map(lambda name: name_score(unit.names, name)),
            gap=(pool.p_nom - unit.capacity).abs(),
        ).query("score >= @min_name_score")
        if pool.empty:
            continue
        best = pool.sort_values(["score", "gap"], ascending=[False, True]).index[0]
        logger.info(
            f"Matched {code} ({unit.names[0]}, {unit.capacity:.0f} MW) by name to "
            f"{ppl.at[best, 'name']} ({ppl.at[best, 'p_nom']:.0f} MW)."
        )
        matches.append(pd.Series(best, index=[code]))
        candidates = candidates.drop(best)
    return pd.concat(matches)


def production_units(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Production units of the outage steps with area, production type, capacity and names.

    The capacity is the nominal power of a production unit outage (``A77``) or
    the sum of the nominal powers of the generation units with outages
    (``A80``), which may be less than the production unit.
    """
    nominal = steps.groupby(["production_resource_id", "unit_name"], dropna=False)
    capacity = nominal.nominal_power.max().groupby(level=0).sum()
    first = steps.groupby("production_resource_id")[["area", "plant_type"]].first()
    names = steps.groupby("production_resource_id").apply(
        lambda s: [
            *s.production_resource_name.dropna().unique(),
            *s.unit_name.dropna().unique(),
        ],
        include_groups=False,
    )
    return first.assign(capacity=capacity, names=names)


def unavailable_share(
    unavailable: pd.DataFrame,
    matches: pd.Series,
    ppl: pd.DataFrame,
    carriers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hourly unavailable share of the capacity per bus and carrier and the share of capacity with outage data.

    The unavailable capacity of a production unit is split between its matched
    plants by their capacity and clipped to the capacity of each plant.

    Parameters
    ----------
    unavailable : pd.DataFrame
        Hourly unavailable capacity in MW per production unit of `hourly_unavailable`.
    matches : pd.Series
        Power plant index per production unit of `match_production_units`.
    ppl : pd.DataFrame
        Power plants of `read_powerplants` with ``area``.
    carriers : list[str]
        Model carriers to include.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Unavailable share with bus and carrier column levels for every bus and
        carrier with capacity, and the capacity, the matched capacity and
        their ratio ``coverage`` per area and carrier.
    """
    ppl = ppl[ppl.carrier.isin(carriers)]
    matches = matches[matches.isin(ppl.index)]
    weight = (
        ppl.p_nom[matches.values].values
        / ppl.p_nom[matches.values].groupby(matches.index).transform("sum").values
    )
    plants = pd.DataFrame(
        unavailable.reindex(columns=matches.index, fill_value=0.0).values * weight,
        index=unavailable.index,
        columns=matches.values,
    )
    plants = plants.T.groupby(level=0).sum().T
    plants = plants.clip(upper=ppl.p_nom[plants.columns], axis=1)
    groups = ppl.loc[plants.columns, ["bus", "carrier"]]
    capacity = ppl.groupby(["bus", "carrier"]).p_nom.sum()
    share = plants.T.groupby([groups.bus, groups.carrier]).sum().T
    share = share.reindex(columns=capacity.index, fill_value=0.0) / capacity
    matched = ppl.index.isin(matches.values)
    coverage = pd.DataFrame(
        {
            "capacity": ppl.groupby(["area", "carrier"]).p_nom.sum(),
            "matched": ppl[matched].groupby(["area", "carrier"]).p_nom.sum(),
        }
    ).fillna(0.0)
    coverage["coverage"] = coverage.matched / coverage.capacity
    return share, coverage


def apply_unavailability(
    n: pypsa.Network,
    share: pd.DataFrame,
    coverage: pd.DataFrame,
    area_to_bus: pd.Series,
    min_coverage: dict[str, float],
    phase_out: dict[str, str],
) -> None:
    """
    Set the availability of thermal generators per bus and carrier to one minus the unavailable share.

    The availability replaces ``p_max_pu`` of all generators of a bus and
    carrier in `share`, including the monthly nuclear availability. Carriers
    of an area whose share of capacity with outage data is below
    `min_coverage` keep their availability. After the last operating date of
    a nuclear area in `phase_out`, its nuclear availability is zero.

    Parameters
    ----------
    n : pypsa.Network
        Network with snapshots.
    share : pd.DataFrame
        Hourly unavailable share with bus and carrier column levels.
    coverage : pd.DataFrame
        Share of capacity with outage data ``coverage`` per area and carrier.
    area_to_bus : pd.Series
        ENTSO-E area per bus.
    min_coverage : dict[str, float]
        Minimum coverage per carrier, zero for carriers not given.
    phase_out : dict[str, str]
        Last operating date of the nuclear generators per area as ISO date.
    """
    assert_known_areas(phase_out, area_to_bus, "the nuclear phase-out")
    share = share.reindex(n.snapshots)
    assert share.notna().all().all(), "Unavailability does not cover all snapshots."
    g = n.generators
    applied = []
    for (bus, carrier), unavailable in share.items():
        gens = g.index[(g.bus == bus) & (g.carrier == carrier)]
        area = area_to_bus.get(bus)
        if gens.empty or area is None:
            continue
        covered = coverage.coverage.get((area, carrier), 0.0)
        if covered < min_coverage.get(carrier, 0.0):
            logger.info(
                f"Keeping the {carrier} availability in {area}, only {covered:.0%} of "
                "its capacity has outage data."
            )
            continue
        available = (1 - unavailable).clip(0, 1)
        if carrier == "nuclear" and area in phase_out:
            end = pd.Timestamp(phase_out[area]) + pd.Timedelta(days=1)
            available = available.where(n.snapshots < end, 0.0)
        n.generators_t.p_max_pu[gens] = np.tile(available.values[:, None], len(gens))
        applied.append((area, carrier, available.mean()))
    summary = pd.DataFrame(applied, columns=["area", "carrier", "availability"])
    logger.info(
        "Mean availability from ENTSO-E unavailability per area and carrier:\n"
        f"{summary.groupby(['area', 'carrier']).availability.mean().unstack().round(2)}"
    )
