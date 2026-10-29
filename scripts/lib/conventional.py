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

logger = logging.getLogger(__name__)


def clip_p_min_pu(n: pypsa.Network) -> None:
    """Clip the minimum to the maximum availability of generators wherever ``p_min_pu`` exceeds ``p_max_pu``."""
    g = n.generators
    gens = g.index[g.p_min_pu > 0].union(n.generators_t.p_min_pu.columns)
    p_min_pu = n.get_switchable_as_dense("Generator", "p_min_pu", inds=gens)
    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu", inds=gens)
    clipped = gens[(p_min_pu > p_max_pu).any()]
    if clipped.empty:
        return
    result = p_min_pu[clipped].clip(upper=p_max_pu[clipped])
    n.generators_t.p_min_pu[clipped] = result
    # The netCDF export drops constant time series, which would restore the static
    # value on import, so constant clipped series also set the static value.
    constant = result.columns[result.nunique() == 1]
    n.generators.loc[constant, "p_min_pu"] = result[constant].iloc[0]
    logger.info(f"Clipped p_min_pu to p_max_pu for {len(clipped)} generators.")


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
