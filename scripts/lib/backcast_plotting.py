# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Shared style, zone conventions and data helpers for the day-ahead price backcast figures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import geopandas as gpd
    import pypsa

T = TypeVar("T")

SHOWCASE_BUSES = ["DE00", "FR00", "ES00", "ITN1", "NOS2", "HU00"]
SEASONS = {"winter (DJF)": [12, 1, 2], "summer (JJA)": [6, 7, 8]}
ZONE_GROUPS = {
    "Central West": ["DE_LU", "FR", "BE", "NL", "AT", "CH"],
    "Central East": ["CZ", "PL", "SK", "HU", "SI", "HR"],
    "South East": ["RO", "BG", "RS", "GR"],
    "Iberia": ["ES", "PT"],
    "Italy": [
        "IT_NORD",
        "IT_CNOR",
        "IT_CSUD",
        "IT_SUD",
        "IT_CALA",
        "IT_ROSN",
        "IT_SICI",
        "IT_SARD",
    ],
    "Nordic": [
        *[f"NO_{i}" for i in range(1, 6)],
        *[f"SE_{i}" for i in range(1, 5)],
        "FI",
        "DK_1",
        "DK_2",
    ],
    "Baltic": ["EE", "LV", "LT"],
    "British Isles": ["IE_SEM", "GB"],
}
ZONE_ORDER = [area for areas in ZONE_GROUPS.values() for area in areas]
CARRIER_ORDER = [
    "nuclear",
    "lignite",
    "coal",
    "gas",
    "oil",
    "biomass",
    "waste",
    "geothermal",
    "hydro",
    "ror",
    "PHS",
    "onwind",
    "offwind",
    "solar",
]
LOAD_SHEDDING = "load shedding"
STACK = [*CARRIER_ORDER, LOAD_SHEDDING]
CARRIER_GROUPS = {
    "CCGT": "gas",
    "OCGT": "gas",
    "offwind-ac": "offwind",
    "offwind-dc": "offwind",
    "offwind-float": "offwind",
    "solar-hsat": "solar",
    "load": LOAD_SHEDDING,
}
COLOR_KEYS = {LOAD_SHEDDING: "load"}
OBSERVED_COLOR = "#000000"
MODEL_COLOR = "#0072B2"
SECOND_COLOR = "#E69F00"
BAND_COLOR = "#BBBBBB"
DIVERGING = "RdBu_r"
SEQUENTIAL = "viridis"
DENSITY = "cividis"
CET_OFFSET = 1
SHOWCASE_WIDTH = 7.2
DPI = 200


def ordered_areas(areas: pd.Index | list[str]) -> list[str]:
    """Sort ENTSO-E areas by the zone order, asserting that all are known."""
    unknown = sorted(set(areas) - set(ZONE_ORDER))
    assert not unknown, f"Areas missing in ZONE_GROUPS: {unknown}"
    return [area for area in ZONE_ORDER if area in set(areas)]


def representative_buses(area_to_bus: pd.Series, observed: pd.Index) -> pd.Series:
    """Map each area with observed prices to its first observed bus, in zone order."""
    areas = area_to_bus[sorted(observed)]
    first = areas[~areas.duplicated()]
    buses = pd.Series(first.index, index=first.values)
    return buses[ordered_areas(buses.index)].rename_axis("area").rename("bus")


def area_prices(prices: pd.DataFrame, buses: pd.Series) -> pd.DataFrame:
    """Select the prices of the representative buses with area columns."""
    return prices[buses.values].set_axis(buses.index, axis=1)


def cet_hour(index: pd.DatetimeIndex) -> pd.Index:
    """Return the hour of day in fixed UTC+1 for naive UTC timestamps."""
    return (index + pd.Timedelta(hours=CET_OFFSET)).hour


def select_weeks(prices: pd.Series) -> dict[str, pd.DatetimeIndex]:
    """Select the full Monday-start week in Oct-Mar with the highest mean price and the week in Apr-Sep containing the lowest hourly price."""
    start = prices.index.normalize() - pd.to_timedelta(prices.index.weekday, unit="D")
    full = prices.groupby(start).count() == 168
    winter_months = start.month.isin([10, 11, 12, 1, 2, 3])
    weekly = prices[winter_months].groupby(start[winter_months]).mean()
    winter = weekly[full[weekly.index]].idxmax()
    summer = start[prices.index.month.isin(range(4, 10))]
    summer = summer[prices[prices.index.month.isin(range(4, 10))].argmin()]
    return {
        name: pd.date_range(week, periods=168, freq="h")
        for name, week in {"winter": winter, "summer": summer}.items()
    }


def group_colors(tech_colors: dict[str, str], groups: list[str]) -> dict[str, str]:
    """Look up the colour of each carrier group, asserting that all exist."""
    keys = {group: COLOR_KEYS.get(group, group) for group in groups}
    missing = [group for group, key in keys.items() if key not in tech_colors]
    assert not missing, f"No tech_colors for carrier groups {missing}"
    return {group: tech_colors[key] for group, key in keys.items()}


def area_generation(n: pypsa.Network, area_to_bus: pd.Series) -> pd.DataFrame:
    """Return the hourly model generation in MW with area and carrier group columns, counting storage units by positive dispatch."""
    assert (n.snapshot_weightings.generators == 1).all(), "Expect hourly snapshots"
    generators = n.generators.assign(group=n.generators.carrier.replace(CARRIER_GROUPS))
    storage = n.storage_units.assign(group=n.storage_units.carrier)
    dispatch = pd.concat([n.generators_t.p, n.storage_units_t.p.clip(lower=0)], axis=1)
    static = pd.concat([generators, storage])[["bus", "group"]]
    keys = [static.bus.map(area_to_bus).rename("area"), static.group.rename("carrier")]
    return dispatch[static.index].T.groupby(keys).sum().T


def area_flows(n: pypsa.Network, area_to_bus: pd.Series) -> pd.DataFrame:
    """Return the hourly net DC flow in MW between areas with columns ``A->B`` oriented alphabetically."""
    links = n.links.query("carrier == 'DC'")
    area0, area1 = links.bus0.map(area_to_bus), links.bus1.map(area_to_bus)
    links = links[area0 != area1]
    area0, area1 = area0[links.index], area1[links.index]
    sign = np.where(area0 < area1, 1.0, -1.0)
    names = [f"{min(a, b)}->{max(a, b)}" for a, b in zip(area0, area1)]
    flows = n.links_t.p0[links.index] * sign
    return flows.T.groupby(names).sum().T


def separate_groups(ax: Axes, areas: list[str], axis: str = "y") -> None:
    """Draw lines between zone groups for areas plotted at positions 0..n-1."""
    group = pd.Series({a: g for g, members in ZONE_GROUPS.items() for a in members})
    breaks = np.flatnonzero(group[areas].to_numpy()[1:] != group[areas].to_numpy()[:-1])
    draw = ax.axhline if axis == "y" else ax.axvline
    for position in breaks + 0.5:
        draw(position, color="0.5", lw=0.6)


def area_heatmap(
    ax: Axes,
    data: pd.DataFrame,
    cmap: str,
    label: str,
    vlim: tuple[float, float],
    annotate: bool = False,
) -> None:
    """Draw a heatmap with areas as rows, group separators and a colour bar."""
    image = ax.imshow(
        data.to_numpy(), aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1]
    )
    ax.set_yticks(range(len(data.index)), data.index)
    ax.set_xticks(range(len(data.columns)), data.columns)
    separate_groups(ax, list(data.index))
    ax.figure.colorbar(image, ax=ax, label=label, shrink=0.8)
    if annotate:
        for (i, j), value in np.ndenumerate(data.to_numpy()):
            if np.isfinite(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.5
                ax.text(
                    j,
                    i,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if dark else "black",
                )


def symmetric_limit(values: pd.DataFrame | pd.Series, quantile: float = 1.0) -> float:
    """Return the quantile of absolute values as symmetric colour limit."""
    return float(np.nanquantile(np.abs(np.asarray(values, dtype=float)), quantile))


def save_figures(
    registry: dict[str, Callable[[T], Figure]], inputs: T, outputs: list[str]
) -> None:
    """Draw each registered figure and write it as PDF and PNG to the outputs with matching stem."""
    stems = {Path(path).stem for path in outputs}
    assert stems == set(registry), f"Figures {set(registry)} differ from {stems}"
    for stem, draw in registry.items():
        fig = draw(inputs)
        for path in outputs:
            if Path(path).stem == stem:
                fig.savefig(path, dpi=DPI)
        plt.close(fig)


@dataclass
class PriceInputs:
    """Aligned bus prices with area views, metrics and selected weeks."""

    model: pd.DataFrame
    observed: pd.DataFrame
    area_to_bus: pd.Series
    buses: pd.Series
    metrics: pd.DataFrame
    monthly_bias: pd.DataFrame
    regions: gpd.GeoDataFrame
    weeks: dict[str, pd.DatetimeIndex] = field(init=False)
    area_model: pd.DataFrame = field(init=False)
    area_observed: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        missing = sorted(set(SHOWCASE_BUSES) - set(self.observed.columns))
        assert not missing, f"Showcase buses without observed prices: {missing}"
        self.area_model = area_prices(self.model, self.buses)
        self.area_observed = area_prices(self.observed, self.buses)
        self.weeks = select_weeks(self.observed[SHOWCASE_BUSES[0]])

    @property
    def area_error(self) -> pd.DataFrame:
        """Hourly model minus observed price per area."""
        return self.area_model - self.area_observed

    def label(self, bus: str) -> str:
        """Return the area code of a bus."""
        return self.area_to_bus[bus]
