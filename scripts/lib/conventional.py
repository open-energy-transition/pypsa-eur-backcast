# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Operational limits and costs of generators applied once the network is composed.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from scripts.lib.backcast_calibration import month_hours, snapshot_months
from scripts.lib.entsoe import assert_known_areas

logger = logging.getLogger(__name__)


def clip_p_min_pu(n: pypsa.Network) -> None:
    """Clip the minimum to the maximum availability of generators wherever ``p_min_pu`` exceeds ``p_max_pu``."""
    g = n.generators
    gens = g.index[g.p_min_pu > 0].union(n.generators_t.p_min_pu.columns)
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=gens)
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", inds=gens)
    clipped = p_min_pu.columns[(p_min_pu > p_max_pu[p_min_pu.columns]).any()]
    if clipped.empty:
        return
    set_switchable(n, "p_min_pu", p_min_pu[clipped].clip(upper=p_max_pu[clipped]))
    logger.info(f"Clipped p_min_pu to p_max_pu for {len(clipped)} generators.")


def set_switchable(n: pypsa.Network, attr: str, values: pd.DataFrame) -> None:
    """Set the time-varying `attr` of the generators in the columns of `values`."""
    n.generators_t[attr] = pd.concat(
        [n.generators_t[attr].drop(columns=values.columns, errors="ignore"), values],
        axis=1,
    )
    constant = values.columns[values.nunique() == 1]
    n.generators.loc[constant, attr] = values[constant].iloc[0]


def set_fuel_price(costs: pd.DataFrame, price: dict[str, float]) -> None:
    """Replace the fuel cost in EUR/MWh_th of the technologies in `price` and their marginal cost."""
    missing = sorted(set(price) - set(costs.index))
    assert not missing, f"No cost data to set the fuel price of: {missing}"
    techs = list(price)
    costs.loc[techs, "fuel"] = pd.Series(price)
    costs.loc[techs, "marginal_cost"] = (
        costs.VOM[techs] + costs.fuel[techs] / costs.efficiency[techs]
    )
    logger.info(f"Set the fuel price of {techs}.")


def observed_carrier(
    generation: pd.DataFrame, carrier: str, snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """Return the observed hourly output in MW of `carrier` per ENTSO-E area over `snapshots`."""
    missing = snapshots.difference(generation.index)
    assert missing.empty, f"Observed generation misses {len(missing)} snapshots."
    return generation.xs(carrier, axis=1, level="carrier").loc[snapshots]


def apply_nuclear_min_output(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    area_map: dict[str, str],
    load_following: dict[str, float],
) -> None:
    """
    Set the nuclear ``p_min_pu`` per ENTSO-E area and month to the lowest observed output.

    The minimum is the lowest hourly output of the month over the nuclear
    capacity of the area, clipped to [0, 1], and zero in months without
    observed output. Generators of the keys of `area_map` are pooled with the
    area they map to. Areas in `load_following` get their share of ``p_max_pu``
    instead. Areas without observed nuclear output keep their ``p_min_pu``.
    """
    assert_known_areas(
        [*area_map, *area_map.values(), *load_following],
        area_to_bus,
        "the nuclear minimum output",
    )
    g = n.generators.query("carrier == 'nuclear'")
    areas = g.bus.map(area_to_bus).replace(area_map)
    observed = observed_carrier(generation, "nuclear", n.snapshots)
    following = g.index[areas.isin(load_following)]
    calibrated = g.index[areas.isin(observed.columns) & ~areas.isin(load_following)]
    capacity = g.p_nom[calibrated].groupby(areas[calibrated]).sum()
    minimum = observed[capacity.index].groupby(snapshot_months(n)).transform("min")
    share = minimum.div(capacity).clip(0, 1).fillna(0.0)
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", inds=following)
    p_min_pu = pd.concat(
        [
            share[areas[calibrated]].set_axis(calibrated, axis=1),
            p_max_pu.mul(areas[following].map(load_following)),
        ],
        axis=1,
    )
    set_switchable(n, "p_min_pu", p_min_pu)
    logger.info(
        f"Set the nuclear minimum output of {sorted(capacity.index)} to the observed "
        f"monthly minimum and of {sorted(load_following)} to load following."
    )


def apply_energy_budget(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    carrier: str,
    exclude: list[str],
    monthly_ceiling: bool,
    reported: pd.Series,
) -> None:
    """
    Cap the energy of `carrier` per ENTSO-E area at the larger of the observed and the `reported` energy.

    The observed energy is scaled per month from the hours with data to all
    hours; every month needs observed data. `reported` holds the annual energy
    in MWh per country, shared by ``p_nom`` over all generators of `carrier` in
    the country; leave it empty to cap at the observed energy. ``e_sum_max`` is
    the area energy shared by ``p_nom`` and raised to the must-run energy of
    ``p_min_pu``. With `monthly_ceiling`, ``p_max_pu`` is clipped to the observed
    hourly peak of the month over the capacity of the area. Areas in `exclude`
    and areas without observed output keep their limits.
    """
    assert_known_areas(exclude, area_to_bus, f"the {carrier} energy budget")
    g = n.generators.query("carrier == @carrier")
    observed = observed_carrier(generation, carrier, n.snapshots)
    areas = g.bus.map(area_to_bus)
    produced = observed.columns[observed.sum() > 0]
    areas = areas[areas.isin(produced) & ~areas.isin(exclude)]
    gens = areas.index
    capacity = g.p_nom[gens].groupby(areas).sum()
    observed = observed[capacity.index]
    weights = n.snapshot_weightings.generators
    months = snapshot_months(n)
    hours = observed.notna().mul(weights, axis=0).groupby(months).sum()
    gaps = hours.columns[(hours == 0).any()]
    assert gaps.empty, (
        f"Observed {carrier} generation misses whole months in {list(gaps)}."
    )
    monthly = observed.mul(weights, axis=0).groupby(months).sum().div(hours)
    energy = monthly.mul(month_hours(n), axis=0).sum()
    if not reported.empty:
        countries = g.bus.map(n.buses.country)
        area_reported = reported_area_energy(g.p_nom, countries, areas, reported)
        energy = np.maximum(energy, area_reported)
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", inds=gens)
    if monthly_ceiling:
        peak = observed.groupby(months).transform("max").div(capacity).clip(upper=1)
        p_max_pu = np.minimum(p_max_pu, peak[areas].set_axis(gens, axis=1))
        set_switchable(n, "p_max_pu", p_max_pu)
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=gens)
    must_run = p_min_pu.clip(upper=p_max_pu).mul(weights, axis=0).sum() * g.p_nom[gens]
    available = p_max_pu.mul(weights, axis=0).sum() * g.p_nom[gens]
    budget = areas.map(energy) * g.p_nom[gens] / areas.map(capacity)
    log_areas(
        (must_run - budget).clip(lower=0).groupby(areas).sum(),
        f"Raised the {carrier} energy budget to the must-run energy in TWh",
    )
    log_areas(
        (budget - available).clip(lower=0).groupby(areas).sum(),
        f"The {carrier} energy budget exceeds the available energy, not binding, by TWh",
    )
    n.generators.loc[gens, "e_sum_max"] = np.maximum(budget, must_run)
    logger.info(
        f"Set the {carrier} energy budget of {sorted(capacity.index)} in TWh: "
        + ", ".join(f"{a} {e / 1e6:.3f}" for a, e in energy.items())
    )


def reported_area_energy(
    p_nom: pd.Series, countries: pd.Series, areas: pd.Series, reported: pd.Series
) -> pd.Series:
    """Return the `reported` energy per country shared by `p_nom` over the generators of the country, summed per area of `areas`."""
    missing = sorted(set(countries[areas.index]) - set(reported.index))
    assert not missing, f"No reported energy for the countries {missing}."
    share = p_nom / p_nom.groupby(countries).transform("sum")
    return (countries.map(reported) * share)[areas.index].groupby(areas).sum()


def log_areas(excess: pd.Series, message: str) -> None:
    """Warn with `message` and the positive `excess` in MWh per area in TWh."""
    excess = excess[excess > 0]
    if not excess.empty:
        logger.warning(
            f"{message}: " + ", ".join(f"{a} {e / 1e6:.3f}" for a, e in excess.items())
        )


CHP_FUEL_CARRIERS = {
    "gas": ["CCGT", "OCGT"],
    "coal": ["coal"],
    "lignite": ["lignite"],
    "oil": ["oil"],
    "biomass": ["biomass"],
    "waste": ["waste"],
}


def apply_chp_floor(
    n: pypsa.Network, floor: pd.DataFrame, replace_p_min_pu: list[str]
) -> None:
    """Raise ``p_min_pu`` to the heat-led CHP floor in MW per country and fuel, shared by capacity over the generators of the group; the flat ``p_min_pu`` of fuels in ``replace_p_min_pu`` is dropped where a floor applies."""
    carrier_fuel = {c: f for f, cs in CHP_FUEL_CARRIERS.items() for c in cs}
    g = n.generators
    groups = pd.MultiIndex.from_arrays(
        [g.bus.map(n.buses.country), g.carrier.map(carrier_fuel)]
    )
    covered = g.index[groups.isin(floor.columns)]
    keys = groups[g.index.get_indexer(covered)]
    floor = floor.reindex(n.snapshots)
    assert floor.notna().all().all(), "CHP floor does not cover the snapshots."
    unmatched = floor.loc[:, ~floor.columns.isin(groups)]
    unmatched = unmatched.mul(n.snapshot_weightings.generators, axis=0).sum()
    unmatched = unmatched[unmatched > 0]
    if not unmatched.empty:
        logger.warning(
            "CHP floor without generators, dropped in TWh: "
            + ", ".join(f"{c} {f} {e / 1e6:.3f}" for (c, f), e in unmatched.items())
        )
    p_nom = g.p_nom[covered].groupby(keys).sum()
    share = floor[keys].div(p_nom[keys].to_numpy()).set_axis(covered, axis=1)
    replaced = covered[keys.get_level_values(1).isin(replace_p_min_pu)]
    n.generators.loc[replaced, "p_min_pu"] = 0.0
    n.generators_t.p_min_pu = n.generators_t.p_min_pu.drop(
        columns=replaced, errors="ignore"
    )
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=covered)
    set_switchable(n, "p_min_pu", np.maximum(p_min_pu, share))
    logger.info(
        f"Applied the CHP floor to {len(covered)} generators in {len(p_nom)} country and fuel groups."
    )


OBSERVED_GROUPS = {
    "CCGT": "gas",
    "OCGT": "gas",
    "coal": "coal",
    "lignite": "lignite",
    "oil": "oil",
}


def observed_reference(
    g: pd.DataFrame,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    utilisation: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Return the reference output and utilisation of the generators `g` from the observed generation.

    Per ENTSO-E area and carrier group, the reference output scales ``p_nom``
    by the observed peak relative to the summed ``p_nom``, capped at 1, and
    the utilisation is the observed mean relative to the observed peak.
    Generators without observed generation or with a zero peak keep ``p_nom``
    and `utilisation`.
    """
    area = g.bus.map(area_to_bus)
    group = g.carrier.map(OBSERVED_GROUPS)
    key = pd.MultiIndex.from_arrays([area, group])
    stats = generation.agg(["max", "mean"]).T.reindex(key).set_axis(g.index)
    capacity = g.p_nom.groupby([area, group]).transform("sum")
    factor = (stats["max"] / capacity).clip(upper=1)
    observed = factor > 0
    p_ref = g.p_nom.where(~observed, g.p_nom * factor)
    observed_utilisation = stats["mean"] / stats["max"]
    return p_ref, observed_utilisation.where(observed, utilisation)


def spread_marginal_cost(
    n: pypsa.Network,
    carriers: list[str],
    spread: float,
    utilisation: float,
    generation: pd.DataFrame | None = None,
    area_to_bus: pd.Series | None = None,
) -> None:
    """
    Let the marginal cost of the generators of `carriers` rise linearly with their output.

    With the mean marginal cost ``mc`` of a generator over all snapshots, the
    marginal cost rises from ``mc * (1 - 2 * utilisation * spread)`` at zero
    output to ``mc * (1 + 2 * (1 - utilisation) * spread)`` at the reference
    output by shifting the linear and setting the quadratic marginal cost.
    The reference output is ``p_nom``, or with the observed hourly
    `generation` per area and carrier group and `area_to_bus` the one of
    `observed_reference`.
    """
    g = n.generators
    missing = sorted(set(carriers) - set(g.carrier))
    assert not missing, f"No generators to spread the marginal cost of: {missing}"
    gens = g.index[g.carrier.isin(carriers)]
    invalid = gens[g.p_nom_extendable[gens] | (g.p_nom[gens] <= 0)]
    assert invalid.empty, (
        f"Marginal cost spread requires fixed positive p_nom: {invalid.tolist()}"
    )
    p_ref, utilisation = (
        (g.p_nom[gens], utilisation)
        if generation is None
        else observed_reference(g.loc[gens], generation, area_to_bus, utilisation)
    )
    mean = n.get_switchable_as_dense("Generator", "marginal_cost", inds=gens).mean()
    offset = 2 * utilisation * spread * mean
    dynamic = gens.intersection(n.generators_t.marginal_cost.columns)
    g.loc[gens, "marginal_cost"] -= offset
    n.generators_t.marginal_cost[dynamic] -= offset[dynamic]
    g.loc[gens, "marginal_cost_quadratic"] = spread * mean / p_ref
    logger.info(
        f"Spread the marginal cost of {len(gens)} generators by {spread} over "
        f"{'the observed peak' if generation is not None else 'p_nom'}."
    )


def cap_fuel_price(fuel_price: pd.DataFrame, cap: dict[str, float]) -> pd.DataFrame:
    """Clip the fuel price per snapshot to the cap in force since the latest ISO date in `cap`, uncapped before the first date."""
    steps = pd.Series(cap, dtype=float).rename(index=pd.Timestamp).sort_index()
    limit = steps.reindex(fuel_price.index, method="ffill").fillna(np.inf)
    return fuel_price.clip(upper=limit, axis=0)


def step_fuel_price(fuel_price: pd.DataFrame, price: dict[str, float]) -> pd.DataFrame:
    """Set the fuel price per snapshot to the price in force since the latest ISO date in `price`, unchanged before the first date."""
    steps = pd.Series(price, dtype=float).rename(index=pd.Timestamp).sort_index()
    level = steps.reindex(fuel_price.index, method="ffill")
    stepped = pd.DataFrame(
        np.repeat(level.to_numpy()[:, None], fuel_price.shape[1], axis=1),
        index=fuel_price.index,
        columns=fuel_price.columns,
    )
    return stepped.fillna(fuel_price)
