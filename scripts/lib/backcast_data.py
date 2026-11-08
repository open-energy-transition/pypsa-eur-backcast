# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Data transforms of the day-ahead price backcast shared by the static figures and the dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scripts.lib.backcast_plotting import (
    CARRIER_GROUPS,
    STACK,
    ZONE_GROUPS,
    cet_hour,
    ordered_areas,
)

if TYPE_CHECKING:
    import pypsa

MIN_COVERAGE = 0.95
MARGINAL_TOLERANCE = 1.0
NO_MARGINAL = "none"
CLIP_PRICE_RANGE = (-100.0, 500.0)
ZERO_PRICE_TOLERANCE = 0.01


def align_prices(
    model: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both prices to common hours and observed buses, masking hours without observed price."""
    assert model.index.tz is None and observed.index.tz is None, (
        "Prices must have naive UTC timestamps"
    )
    missing = observed.columns.difference(model.columns)
    assert missing.empty, f"Observed buses missing in model: {missing.tolist()}"
    hours = model.index.intersection(observed.index)
    assert not hours.empty, "Modelled and observed prices share no hours"
    observed = observed.loc[hours]
    model = model.loc[hours, observed.columns].where(observed.notna())
    return model, observed


def share(mask: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Share of hours with valid price fulfilling the mask."""
    return mask.astype(float).where(prices.notna()).mean()


def rmse(model: pd.DataFrame, observed: pd.DataFrame) -> pd.Series:
    """Root mean squared error per bus."""
    return np.sqrt(((model - observed) ** 2).mean())


def price_metrics(model: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute error metrics and zero and negative price shares per bus."""
    error = model - observed
    metrics = {
        "mean_bias": error.mean(),
        "mae": error.abs().mean(),
        "rmse": rmse(model, observed),
        "rmse_clipped": rmse(
            model.clip(*CLIP_PRICE_RANGE), observed.clip(*CLIP_PRICE_RANGE)
        ),
        "pearson_r": model.corrwith(observed),
        "std_ratio": model.std() / observed.std(),
        "mean_observed": observed.mean(),
        "mean_model": model.mean(),
        "std_observed": observed.std(),
    }
    for name, prices in {"model": model, "observed": observed}.items():
        metrics[f"zero_share_{name}"] = share(
            prices.abs() <= ZERO_PRICE_TOLERANCE, prices
        )
        metrics[f"negative_share_{name}"] = share(
            prices < -ZERO_PRICE_TOLERANCE, prices
        )
    return pd.DataFrame(metrics).rename_axis("bus")


def area_metrics(metrics: pd.DataFrame, area_to_bus: pd.Series) -> pd.DataFrame:
    """Average per-bus rows over the buses of each price area, so that each area counts once."""
    return metrics.groupby(area_to_bus[metrics.index].rename("area")).mean()


def summary_metrics(metrics: pd.DataFrame) -> pd.Series:
    """Summarise per-bus or per-area metrics into the headline numbers of the acceptance rule."""
    at_or_below_zero = {
        f"share_le0_{name}": (
            metrics[f"zero_share_{name}"] + metrics[f"negative_share_{name}"]
        ).mean()
        for name in ["model", "observed"]
    }
    return pd.Series(
        {
            "median_mae": metrics.mae.median(),
            "mean_mae": metrics.mae.mean(),
            "pearson_r": metrics.pearson_r.median(),
            "abs_bias": metrics.mean_bias.abs().median(),
            "std_ratio": metrics.std_ratio.median(),
            **at_or_below_zero,
            "count": len(metrics),
        }
    )


def zone_group_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute the median MAE, bias and correlation per zone group from per-area metrics."""
    group = pd.Series(
        {area: name for name, areas in ZONE_GROUPS.items() for area in areas}
    )
    areas = ordered_areas(metrics.index)
    grouped = metrics.loc[areas, ["mae", "mean_bias", "pearson_r"]].groupby(
        group[areas].rename("group"), sort=False
    )
    return grouped.median().assign(count=grouped.size())


def monthly_bias(model: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean bias per year-month and bus."""
    error = model - observed
    return error.groupby(error.index.to_period("M")).mean().rename_axis("month")


def duration_curve(prices: pd.Series) -> pd.Series:
    """Sort prices descending over the share of hours."""
    values = prices.dropna().sort_values(ascending=False).to_numpy()
    return pd.Series(values, index=np.arange(1, len(values) + 1) / len(values))


def monthly_mae(error: pd.DataFrame) -> pd.DataFrame:
    """Compute the monthly mean absolute error per column."""
    return error.abs().groupby(error.index.to_period("M")).mean()


def decile_errors(error: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean error per area and observed price decile of that area."""
    deciles = observed.rank(pct=True).mul(10).apply(np.ceil).clip(1, 10)
    return pd.DataFrame(
        {area: error[area].groupby(deciles[area]).mean() for area in error}
    ).T.set_axis(range(1, 11), axis=1)


def top_errors(error: pd.DataFrame, observed: pd.DataFrame, top: float) -> pd.Series:
    """Compute the mean error per area in the hours with the highest observed prices."""
    return error.where(observed.ge(observed.quantile(1 - top))).mean()


def hourly_errors(error: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Compute the mean error per area and hour of day (CET) in the months."""
    selected = error[error.index.month.isin(months)]
    return selected.groupby(cet_hour(selected.index)).mean().T


def hourly_area_bias(
    model: pd.DataFrame, observed: pd.DataFrame, area_to_bus: pd.Series
) -> pd.Series:
    """Compute the median over price areas of the mean error per hour of day (CET)."""
    errors = hourly_errors(model - observed, list(range(1, 13)))
    return area_metrics(errors, area_to_bus).median().rename_axis("hour").rename("bias")


def stacked_generation(
    generation: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Select the hourly generation of one area in GW with the carrier stack as columns."""
    return generation.reindex(index=index, columns=STACK).fillna(0.0) / 1e3


def net_exchanges(exchanges: pd.DataFrame) -> pd.DataFrame:
    """Net scheduled exchanges per alphabetically oriented area border with sufficient coverage of both directions."""
    pairs = {tuple(sorted(c.split("->"))) for c in exchanges.columns}
    net = {}
    for a, b in sorted(pairs):
        forward = exchanges.get(f"{a}->{b}", pd.Series(np.nan, exchanges.index))
        backward = exchanges.get(f"{b}->{a}", pd.Series(np.nan, exchanges.index))
        flow = forward - backward
        if flow.notna().mean() >= MIN_COVERAGE:
            net[f"{a}->{b}"] = flow
    return pd.DataFrame(net, index=exchanges.index)


def area_capacities(capacities: pd.DataFrame, area_to_bus: pd.Series) -> pd.DataFrame:
    """Sum directed bus-to-bus transfer capacities to directed area borders ``A->B``, dropping intra-area pairs."""
    pairs = capacities.columns.str.split("->", expand=True)
    area0 = pairs.get_level_values(0).map(area_to_bus)
    area1 = pairs.get_level_values(1).map(area_to_bus)
    keep = (area0 != area1) & area0.notna() & area1.notna()
    names = [f"{a}->{b}" for a, b in zip(area0[keep], area1[keep])]
    return capacities.loc[:, keep].T.groupby(names).sum().T


def closest_group(
    cost: np.ndarray, groups: np.ndarray, price: np.ndarray, tolerance: float
) -> np.ndarray:
    """Return per hour the group of the column with cost closest to the price within the tolerance, NaN cost marking units not dispatching."""
    distance = np.nan_to_num(np.abs(cost - price[:, None]), nan=np.inf)
    best = distance.argmin(axis=1)
    found = distance[np.arange(len(price)), best] <= tolerance
    return np.where(found, groups[best], NO_MARGINAL)


def marginal_carriers(
    n: pypsa.Network, tolerance: float = MARGINAL_TOLERANCE
) -> pd.DataFrame:
    """Assign each AC bus and hour the carrier group of the dispatching generator with marginal cost closest to the bus price within the tolerance, preferring generators at the bus over all others."""
    buses = n.buses.query("carrier == 'AC'").index
    prices = n.buses_t.marginal_price[buses]
    generators = n.generators
    cost = n.get_switchable_as_dense("Generator", "marginal_cost")[generators.index]
    cost = cost.where(n.generators_t.p[generators.index] > 1e-3).to_numpy()
    groups = generators.carrier.replace(CARRIER_GROUPS).to_numpy()
    carriers = {}
    for bus in buses:
        price = prices[bus].to_numpy()
        local = (generators.bus == bus).to_numpy()
        found = closest_group(cost[:, local], groups[local], price, tolerance)
        anywhere = closest_group(cost, groups, price, tolerance)
        carriers[bus] = np.where(found == NO_MARGINAL, anywhere, found)
    return pd.DataFrame(carriers, index=prices.index)
