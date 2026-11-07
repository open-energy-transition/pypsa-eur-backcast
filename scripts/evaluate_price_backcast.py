# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Evaluates modelled against observed hourly day-ahead electricity prices per bus.

The modelled prices are the marginal prices of the AC buses of the solved
network, the observed prices are built by
:mod:`build_day_ahead_prices`. Both are compared on their common naive UTC
hours for the buses with observed prices, skipping hours without observed
price. The per-bus metrics are the mean bias, mean absolute error, root mean
squared error, root mean squared error of prices clipped to
``CLIP_PRICE_RANGE``, Pearson correlation, ratio of standard deviations, the
observed and modelled mean prices, the observed standard deviation and the
shares of zero and negative price hours. The same metrics per price area,
averaging the buses of one ENTSO-E area so that each area counts once, and the
monthly mean bias are written as separate tables. Monthly means, price
duration curves, average daily profiles and hourly scatter plots are written
to one multi-page PDF.
"""

import logging
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lib.backcast_data import (
    align_prices,
    area_metrics,
    duration_curve,
    monthly_bias,
    price_metrics,
)
from scripts.lib.entsoe import read_area_to_bus

logger = logging.getLogger(__name__)

PANEL_COLUMNS = 6

Draw = Callable[[Axes, pd.Series, pd.Series], None]


def draw_lines(transform: Callable[[pd.Series], pd.Series]) -> Draw:
    """Return a drawer plotting the transformed modelled and observed prices as lines."""

    def draw(ax: Axes, model: pd.Series, observed: pd.Series) -> None:
        transform(model).plot(ax=ax, label="model")
        transform(observed).plot(ax=ax, label="observed", color="k", alpha=0.7)

    return draw


def draw_scatter(ax: Axes, model: pd.Series, observed: pd.Series) -> None:
    """Scatter modelled over observed hourly prices with the identity line."""
    ax.scatter(observed, model, s=1, alpha=0.2, linewidths=0, rasterized=True)
    bounds = [np.nanmin([observed, model]), np.nanmax([observed, model])]
    ax.plot(bounds, bounds, color="k", lw=0.8)
    ax.set_xlabel("observed €/MWh")
    ax.set_ylabel("model €/MWh")


PLOTS: dict[str, Draw] = {
    "Monthly mean price": draw_lines(lambda s: s.resample("MS").mean()),
    "Price duration curve": draw_lines(duration_curve),
    "Average daily profile (UTC hour)": draw_lines(
        lambda s: s.groupby(s.index.hour).mean()
    ),
    "Hourly prices": draw_scatter,
}


def plot_page(
    pdf: PdfPages, title: str, draw: Draw, model: pd.DataFrame, observed: pd.DataFrame
) -> None:
    """Add one page with a panel per bus to the PDF."""
    rows = -(-len(observed.columns) // PANEL_COLUMNS)
    fig, axes = plt.subplots(
        rows, PANEL_COLUMNS, figsize=(3 * PANEL_COLUMNS, 2.5 * rows), squeeze=False
    )
    for ax, bus in zip(axes.flat, observed.columns):
        ax.set_ylabel("€/MWh")
        draw(ax, model[bus], observed[bus])
        ax.set_title(bus)
    for ax in axes.flat[len(observed.columns) :]:
        ax.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("evaluate_price_backcast")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    model = n.buses_t.marginal_price[n.buses.query("carrier == 'AC'").index]
    observed = pd.read_csv(snakemake.input.prices, index_col=0, parse_dates=True)
    model, observed = align_prices(model, observed)

    metrics = price_metrics(model, observed)
    logger.info(f"Price backcast metrics:\n{metrics.round(2)}")
    metrics.to_csv(snakemake.output.metrics)
    area_to_bus = read_area_to_bus(snakemake.input.area_to_bus)
    area_metrics(metrics, area_to_bus).to_csv(snakemake.output.area_metrics)
    monthly_bias(model, observed).to_csv(snakemake.output.monthly_bias)

    with PdfPages(snakemake.output.plots) as pdf:
        for title, draw in PLOTS.items():
            plot_page(pdf, title, draw, model, observed)
