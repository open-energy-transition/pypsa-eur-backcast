# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
How it works: method, headline results, scoring, limitations and refinement history, and the run facts filling the prose.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import narrative
from dashboard.data import (
    CACHE,
    EXPORT_MODEL,
    EXPORT_SCHEDULED,
    FOCUS_AREA,
    TOP_ERROR,
    TOP_SHARE,
    RunData,
    area_metrics,
    history_frame,
    load_prices,
    summary,
)
from dashboard.views.common import CHART_HEIGHT, Context, export_direction, worst
from scripts.lib.backcast_plotting import MODEL_COLOR, SECOND_COLOR

INTEGER_ROWS = ["year", "scored model regions"]
YEAR_COLORS = {"2023": MODEL_COLOR, "2019": SECOND_COLOR}


def facts(run: RunData, numbers: dict[str, float]) -> dict[str, str]:
    """Values filling the placeholders of the prose for a run."""
    metrics = area_metrics(run.area_model, run.area_observed)
    model = numbers[EXPORT_MODEL]
    scheduled = numbers[EXPORT_SCHEDULED]
    return {
        "run": run.name,
        "r": f"{numbers['median Pearson r']:.2f}",
        "top": f"{TOP_SHARE:.0%}",
        "top_error": f"{numbers[TOP_ERROR]:+.0f}",
        "nonpositive_observed": f"{numbers['hours ≤ 0 observed (%)']:.1f} %",
        "nonpositive_model": f"{numbers['hours ≤ 0 model (%)']:.1f} %",
        "exporter": narrative.NAMES[FOCUS_AREA],
        "direction": export_direction(model, scheduled),
        "export_model": f"{model:+.0f}",
        "export_scheduled": f"{scheduled:+.0f}",
        "high_bias": worst(metrics.mean_bias),
        "low_bias": worst(-metrics.mean_bias),
        "worst_mae": worst(metrics.mae),
    }


def pipeline() -> None:
    """Draw inputs, model and output as three boxes."""
    for column, (title, items) in zip(st.columns(3), narrative.PIPELINE.items()):
        box = column.container(border=True)
        box.markdown(f"**{title}**")
        box.markdown("\n".join(f"- {item}" for item in items))


def method() -> None:
    """Draw what the backcast is and how it works."""
    st.header("What the model does")
    st.markdown(narrative.WHAT)
    st.markdown(narrative.HOW)
    pipeline()


def headline_results(table: pd.DataFrame, filled: dict[str, str]) -> None:
    """Draw what works and the whole-year results per run in an expander."""
    st.header("What works")
    st.markdown(narrative.WORKS.format(**filled))
    with st.expander("Headline results per run"):
        st.caption(
            "Whole year per run, independent of the period in the sidebar. "
            f"Top {TOP_SHARE:.0%} hours are the most expensive observed hours of each model region."
        )
        shown = table.map("{:.2f}".format)
        shown.loc[INTEGER_ROWS] = table.loc[INTEGER_ROWS].map("{:.0f}".format)
        st.table(shown)


def scoring() -> None:
    """Draw how a run is scored."""
    st.header("How it is scored")
    st.markdown(narrative.SCORING)


def limitations(filled: dict[str, str]) -> None:
    """Draw the limitations table filled from the run."""
    st.header("What it leaves out")
    st.markdown(narrative.LIMITATIONS.format(**filled))


def history_chart(frame: pd.DataFrame) -> go.Figure:
    """Median MAE per refinement step and year as a step line through the kept steps, kept steps filled and rejected ones hollow."""
    fig = go.Figure()
    for year, color in YEAR_COLORS.items():
        kept = frame[frame.kept].dropna(subset=year)
        fig.add_scatter(
            x=kept.index,
            y=kept[year],
            mode="lines",
            line=dict(color=color, shape="hv"),
            name=f"kept model {year}",
            hoverinfo="skip",
        )
        steps = frame.dropna(subset=year)
        fig.add_scatter(
            x=steps.index,
            y=steps[year],
            mode="markers",
            marker=dict(
                color=color,
                size=10,
                symbol=np.where(steps.kept, "circle", "circle-open"),
                line_width=2,
            ),
            name=year,
            showlegend=False,
            customdata=steps[["step", "verdict"]].to_numpy(),
            hovertemplate=f"{year}: %{{y:.1f}} €/MWh<br>%{{customdata[0]}}<br>%{{customdata[1]}}<extra></extra>",
        )
    fig.update_xaxes(title="step", dtick=1)
    fig.update_layout(
        height=CHART_HEIGHT,
        yaxis_title="median MAE (€/MWh)",
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def history() -> None:
    """Draw the refinement history as a chart with the recorded table in an expander."""
    st.header("How the model got better")
    st.markdown(narrative.HISTORY_LEAD)
    frame = history_frame(narrative.HISTORY)
    st.plotly_chart(history_chart(frame), width="stretch")
    with st.expander("Every step as recorded (median MAE in €/MWh, 2023 / 2019)"):
        st.table(frame[["step", "result", "verdict"]])


@st.cache_data
def run_summary(name: str) -> dict[str, float]:
    """Headline numbers of a cached run, read from its prices and flows only."""
    path = CACHE / name
    return summary(load_prices(path), pd.read_parquet(path / "flows.parquet"))


def render(ctx: Context) -> None:
    """Draw the how-it-works page from method to refinement history, with whole-year results of the run and the cached runs of other years."""
    names = [ctx.run.name, *[ref.run.name for ref in ctx.references]]
    table = pd.DataFrame({name: run_summary(name) for name in names})
    filled = facts(ctx.run, table[ctx.run.name].to_dict())
    method()
    headline_results(table, filled)
    scoring()
    limitations(filled)
    history()
