# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plots the benchmark figures of the modelled against the observed generation, exchanges and price coupling.

Annual and monthly generation per area and carrier are compared to the ENTSO-E
actual generation, storage units counted by positive dispatch. Hourly price
correlations between areas and price spreads across the DC borders show the
market coupling, the net DC flows are compared to the ENTSO-E scheduled
commercial exchanges of borders with at least ``MIN_COVERAGE`` published
hours. The hourly dispatch in the winter and summer week of the price figures
is compared for ``DISPATCH_AREAS``. Every figure is written as PDF and PNG.
"""

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.backcast_data import (
    MIN_COVERAGE,
    align_prices,
    net_exchanges,
    stacked_generation,
)
from scripts.lib.backcast_plotting import (
    DIVERGING,
    LOAD_SHEDDING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEQUENTIAL,
    STACK,
    area_flows,
    area_generation,
    area_heatmap,
    area_prices,
    group_colors,
    representative_buses,
    save_figures,
    select_weeks,
    separate_groups,
    symmetric_limit,
)
from scripts.lib.entsoe import read_area_to_bus, read_hourly_generation

DECOUPLING_SPREAD = 1.0
OUTLIER_LABELS = 10
DISPATCH_AREAS = ["DE_LU", "FR", "ES"]
MIX_GROUPS = {
    "nuclear": ["nuclear"],
    "lignite+coal": ["lignite", "coal"],
    "gas": ["gas"],
    "hydro+ror+PHS": ["hydro", "ror", "PHS"],
    "wind": ["onwind", "offwind"],
    "solar": ["solar"],
}
MIX_COLOR_KEYS = {group: members[0] for group, members in MIX_GROUPS.items()}


@dataclass
class SystemInputs:
    """Model and ENTSO-E generation, prices and flows per area."""

    generation: pd.DataFrame
    observed_monthly: pd.DataFrame
    observed_hourly: pd.DataFrame
    model_prices: pd.DataFrame
    observed_prices: pd.DataFrame
    flows: pd.DataFrame
    exchanges: pd.DataFrame
    colors: dict[str, str]

    @property
    def areas(self) -> list[str]:
        """Areas with observed prices and generation in zone order."""
        observed = set(self.observed_monthly.index.get_level_values("area"))
        return [a for a in self.observed_prices.columns if a in observed]


def annual_mix(d: SystemInputs) -> dict[str, pd.DataFrame]:
    """Annual generation in TWh per area and carrier for model and ENTSO-E."""
    model = d.generation.sum().unstack("carrier").reindex(d.areas) / 1e6
    observed = d.observed_monthly.generation.groupby(["area", "carrier"]).sum()
    observed = observed.unstack("carrier").reindex(d.areas) / 1e6
    return {
        name: df.reindex(columns=STACK).fillna(0.0)
        for name, df in {"model": model, "ENTSO-E": observed}.items()
    }


def annual_hatch(carrier: str, name: str) -> str | None:
    """Hatch bars of load shedding and of the ENTSO-E series."""
    hatch = ("////" if carrier == LOAD_SHEDDING else "") + (
        "" if name == "model" else "...."
    )
    return hatch or None


def generation_annual(d: SystemInputs) -> Figure:
    """Plot stacked annual generation per area for the model and ENTSO-E."""
    mix = annual_mix(d)
    y = np.arange(len(d.areas))
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(d.areas) + 1.5))
    for offset, (name, df) in zip([-0.2, 0.2], mix.items()):
        left = np.zeros(len(y))
        for carrier in STACK:
            ax.barh(
                y + offset,
                df[carrier],
                left=left,
                height=0.38,
                color=d.colors[carrier],
                hatch=annual_hatch(carrier, name),
                hatchcolor="white",
                lw=0,
                label=carrier if name == "model" else None,
            )
            left += df[carrier].to_numpy()
        for yi, total in zip(y, left):
            ax.text(total, yi + offset, f" {name[0].upper()}", va="center", fontsize=5)
    ax.set_yticks(y, d.areas, fontsize=6)
    ax.set_ylim(len(y) - 0.5, -0.5)
    separate_groups(ax, d.areas)
    ax.set_xlabel("annual generation (TWh), M = model (upper), E = ENTSO-E (lower)")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        plt.Rectangle(
            (0, 0), 1, 1, fc="0.5", hatch="....", hatchcolor="white", label="ENTSO-E"
        )
    )
    ax.legend(
        handles=handles,
        ncol=8,
        fontsize=6,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    return fig


def mix_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Sum carrier columns to the carrier groups of the monthly mix."""
    return pd.DataFrame(
        {
            g: df.reindex(columns=m).sum(axis=1, min_count=1)
            for g, m in MIX_GROUPS.items()
        }
    )


def generation_monthly(d: SystemInputs) -> Figure:
    """Plot monthly generation per area and carrier group for the model and ENTSO-E."""
    colors = {g: d.colors[c] for g, c in MIX_COLOR_KEYS.items()}
    model = d.generation.resample("MS").sum() / 1e6
    observed = d.observed_monthly.generation.unstack(["area", "carrier"]) / 1e6
    hours = d.observed_monthly.hours.unstack(["area", "carrier"])
    month_hours = hours.index.days_in_month * 24
    cols = 6
    rows = -(-len(d.areas) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.1 * rows), squeeze=False)
    for ax, area in zip(axes.flat, d.areas):
        modelled = mix_groups(model[area])
        actual = mix_groups(observed[area])
        complete = hours[area].ge(0.9 * month_hours, axis=0)
        for group, members in MIX_GROUPS.items():
            color = colors[group]
            if modelled[group].abs().sum() > 1e-3:
                ax.plot(modelled[group], color=color, lw=1)
            if actual[group].notna().any():
                ax.plot(actual[group], color=color, lw=1, ls="--")
                reported = complete.columns.intersection(members)
                gaps = ~complete[reported].all(axis=1) & actual[group].notna()
                ax.scatter(
                    actual.index[gaps],
                    actual[group][gaps],
                    facecolors="none",
                    edgecolors=color,
                    s=12,
                )
        ax.set_title(area, fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("TWh", fontsize=7)
    for ax in axes.flat[len(d.areas) :]:
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=c, label=g) for g, c in colors.items()]
    handles += [
        plt.Line2D([], [], color="0.3", label="model"),
        plt.Line2D([], [], color="0.3", ls="--", label="ENTSO-E"),
        plt.Line2D(
            [], [], color="0.3", ls="", marker="o", mfc="none", label="<90 % data"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def generation_error(d: SystemInputs) -> Figure:
    """Plot the annual generation error per area and carrier in TWh and percent of the observed total."""
    mix = annual_mix(d)
    error = mix["model"] - mix["ENTSO-E"]
    relative = error.div(mix["ENTSO-E"].sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.2 * len(d.areas) + 1.5))
    for ax, data, label in [
        (axes[0], error, "model − ENTSO-E (TWh)"),
        (axes[1], relative, "model − ENTSO-E (% of observed total)"),
    ]:
        limit = symmetric_limit(data, 0.98)
        area_heatmap(ax, data, DIVERGING, label, (-limit, limit), True)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    return fig


def price_correlation(d: SystemInputs) -> Figure:
    """Plot hourly price correlations between areas, observed below and model above the diagonal."""
    observed, model = d.observed_prices.corr(), d.model_prices.corr()
    lower = np.tril(np.ones(observed.shape, dtype=bool), -1)
    upper = np.triu(np.ones(observed.shape, dtype=bool), 1)
    combined = observed.where(lower, model.where(upper))
    fig, ax = plt.subplots(figsize=(9, 8))
    area_heatmap(ax, combined, SEQUENTIAL, "Pearson r of hourly prices", (0, 1))
    separate_groups(ax, list(combined.columns), axis="x")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("lower: observed, upper: model", fontsize=8)
    fig.tight_layout()
    return fig


def border_spreads(d: SystemInputs) -> pd.DataFrame:
    """Compute spread statistics per DC area border with observed prices on both ends."""
    borders = [b.split("->") for b in d.flows.columns]
    rows = {}
    for a, b in borders:
        if a in d.observed_prices and b in d.observed_prices:
            spreads = {
                name: prices[a] - prices[b]
                for name, prices in {
                    "model": d.model_prices,
                    "observed": d.observed_prices,
                }.items()
            }
            rows[f"{a}-{b}"] = {
                **{f"mean_{n}": s.abs().mean() for n, s in spreads.items()},
                **{
                    f"decoupled_{n}": (s.abs() > DECOUPLING_SPREAD)
                    .where(s.notna())
                    .mean()
                    for n, s in spreads.items()
                },
                "r": spreads["model"].corr(spreads["observed"]),
            }
    return pd.DataFrame(rows).T


def identity_scatter(
    ax: Axes, x: pd.Series, y: pd.Series, color: pd.Series, labels: int
) -> plt.cm.ScalarMappable:
    """Scatter y over x coloured by correlation with identity line and labels of the largest deviations."""
    points = ax.scatter(
        x, y, c=color, cmap=SEQUENTIAL, vmin=0, vmax=1, s=18, edgecolors="k", lw=0.3
    )
    bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(bounds, bounds, color="k", lw=0.6)
    for name in (y - x).abs().nlargest(labels).index:
        ax.annotate(
            name,
            (x[name], y[name]),
            fontsize=5,
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax.tick_params(labelsize=6)
    return points


def price_spreads(d: SystemInputs) -> Figure:
    """Plot modelled against observed price spreads and decoupling shares per DC area border."""
    stats = border_spreads(d)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, column, label in [
        (axes[0], "mean", "mean absolute hourly spread (€/MWh)"),
        (
            axes[1],
            "decoupled",
            f"share of hours with spread > {DECOUPLING_SPREAD:.0f} €/MWh",
        ),
    ]:
        points = identity_scatter(
            ax,
            stats[f"{column}_observed"],
            stats[f"{column}_model"],
            stats.r,
            OUTLIER_LABELS,
        )
        ax.set_xlabel(f"observed {label}", fontsize=7)
        ax.set_ylabel(f"model {label}", fontsize=7)
    fig.colorbar(points, ax=axes, label="Pearson r of hourly spreads", shrink=0.8)
    return fig


def area_net_export(flows: pd.DataFrame, areas: list[str]) -> pd.Series:
    """Annual net export in TWh per area from alphabetically oriented border flows."""
    totals = flows.sum() / 1e6
    export = pd.Series(0.0, index=areas)
    for border, total in totals.items():
        a, b = border.split("->")
        export[a] = export.get(a, 0.0) + total
        export[b] = export.get(b, 0.0) - total
    return export[areas]


def exchanges(d: SystemInputs) -> Figure:
    """Plot modelled against scheduled net flows per area border and net exports per area."""
    scheduled = net_exchanges(d.exchanges)
    covered = d.flows.columns.intersection(scheduled.columns)
    model, observed = d.flows[covered], scheduled[covered]
    valid = observed.notna()
    r = model.where(valid).corrwith(observed)
    border_areas = {area for c in d.flows.columns for area in c.split("->")}
    full = [
        a
        for a in d.observed_prices.columns
        if a in border_areas
        and all(c in covered for c in d.flows.columns if a in c.split("->"))
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, max(4.8, 0.22 * len(full) + 1.5)),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    points = identity_scatter(
        axes[0],
        observed.mean() / 1e3,
        model.where(valid).mean() / 1e3,
        r,
        OUTLIER_LABELS,
    )
    axes[0].set_xlabel("scheduled annual mean net flow A→B (GW)", fontsize=7)
    axes[0].set_ylabel("model annual mean net flow A→B (GW)", fontsize=7)
    axes[0].set_title(
        f"{len(covered)} borders with ≥{MIN_COVERAGE:.0%} coverage", fontsize=8
    )
    fig.colorbar(points, ax=axes[0], label="Pearson r of hourly flows", shrink=0.8)
    exports = {
        "scheduled": area_net_export(observed.fillna(0.0), full),
        "model": area_net_export(model.where(valid, 0.0), full),
    }
    y = np.arange(len(full))
    axes[1].hlines(y, exports["scheduled"], exports["model"], color="0.7")
    axes[1].scatter(
        exports["scheduled"], y, color=OBSERVED_COLOR, s=10, label="scheduled"
    )
    axes[1].scatter(exports["model"], y, color=MODEL_COLOR, s=10, label="model")
    axes[1].axvline(0, color="0.6", lw=0.5)
    axes[1].set_yticks(y, full, fontsize=6)
    axes[1].set_ylim(len(full) - 0.5, -0.5)
    separate_groups(axes[1], full)
    axes[1].set_xlabel("annual net export (TWh)", fontsize=7)
    axes[1].legend(fontsize=6)
    axes[1].tick_params(labelsize=6)
    return fig


def dispatch_weeks(d: SystemInputs) -> Figure:
    """Plot stacked hourly generation of model and ENTSO-E with prices in the winter and summer week."""
    weeks = select_weeks(d.observed_prices[DISPATCH_AREAS[0]])
    fig, axes = plt.subplots(
        2 * len(DISPATCH_AREAS),
        2,
        figsize=(12, 3.2 * len(DISPATCH_AREAS)),
        sharey="row",
    )
    for i, area in enumerate(DISPATCH_AREAS):
        sources = {
            "model": (d.generation[area], d.model_prices[area], MODEL_COLOR),
            "ENTSO-E": (
                d.observed_hourly[area],
                d.observed_prices[area],
                OBSERVED_COLOR,
            ),
        }
        for j, (name, (generation, price, color)) in enumerate(sources.items()):
            for k, week in enumerate(weeks.values()):
                ax = axes[2 * i + j, k]
                stack = stacked_generation(generation, week)
                ax.stackplot(
                    week, stack.T.to_numpy(), colors=[d.colors[c] for c in STACK], lw=0
                )
                twin = ax.twinx()
                twin.plot(week, price[week], color=color, lw=0.8)
                both = pd.concat([d.model_prices[area], d.observed_prices[area]])
                twin.set_ylim(both[week].min() - 10, both[week].max() + 10)
                twin.set_ylabel("€/MWh", fontsize=6)
                twin.tick_params(labelsize=6)
                ax.set_title(f"{area} {name}, week from {week[0]:%d %b}", fontsize=7)
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
                ax.tick_params(labelsize=6)
            axes[2 * i + j, 0].set_ylabel("GW", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=d.colors[c], label=c) for c in STACK]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


FIGURES: dict[str, Callable[[SystemInputs], Figure]] = {
    "generation_annual": generation_annual,
    "generation_monthly": generation_monthly,
    "generation_error": generation_error,
    "price_correlation": price_correlation,
    "price_spreads": price_spreads,
    "exchanges": exchanges,
    "dispatch_weeks": dispatch_weeks,
}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("plot_backcast_system")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    plt.style.use(["default", snakemake.input.rc])

    n = pypsa.Network(snakemake.input.network)
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    model = n.buses_t.marginal_price[n.buses.query("carrier == 'AC'").index]
    observed = pd.read_csv(snakemake.input.prices, index_col=0, parse_dates=True)
    model, observed = align_prices(model, observed)
    buses = representative_buses(area_to_bus, observed.columns)
    monthly = pd.read_csv(
        snakemake.input.generation, index_col=[0, 1, 2], parse_dates=["month"]
    )
    hourly = read_hourly_generation(snakemake.input.generation_hourly)
    inputs = SystemInputs(
        generation=area_generation(n, area_to_bus),
        observed_monthly=monthly,
        observed_hourly=hourly,
        model_prices=area_prices(model, buses),
        observed_prices=area_prices(observed, buses),
        flows=area_flows(n, area_to_bus),
        exchanges=pd.read_csv(snakemake.input.exchanges, index_col=0, parse_dates=True),
        colors=group_colors(
            snakemake.params.tech_colors, [*STACK, *MIX_COLOR_KEYS.values()]
        ),
    )
    save_figures(FIGURES, inputs, snakemake.output)
