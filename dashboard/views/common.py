# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Page context and Plotly helpers shared by the dashboard pages.
"""

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import RunData, RunPrices, area_metrics, bus_metrics
from dashboard.narrative import NAMES
from scripts.lib.backcast_plotting import (
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEQUENTIAL,
    ZONE_GROUPS,
    ZONE_ORDER,
    symmetric_limit,
)

ROW_HEIGHT = 16
CHART_HEIGHT = 420
MAP_HEIGHT = 520
GEOS = dict(
    visible=False,
    projection_type="mercator",
    lonaxis_range=[-10.5, 31.5],
    lataxis_range=[35.5, 71],
)
WORSE_COLOR = "#D55E00"
MONTHS = list(calendar.month_abbr)[1:]
MARKER_COLORS = {"negative price": "#CC79A7", "load shedding": WORSE_COLOR}


@dataclass(frozen=True)
class Line:
    """Price series of one source drawn in the period figures."""

    name: str
    prices: pd.Series
    color: str


@dataclass(frozen=True)
class Reference:
    """Prices of a run of another year over the comparable period."""

    run: RunPrices
    hours: pd.DatetimeIndex
    period: str

    def bus_metrics(self) -> pd.DataFrame:
        """Bus metrics of the run in the period."""
        return bus_metrics(self.run.model, self.run.observed, self.hours)


@dataclass(frozen=True)
class Context:
    """Run and the sidebar selection."""

    run: RunData
    area: str
    hours: pd.DatetimeIndex
    period: str
    aggregation: str
    colors: dict[str, str]
    references: list[Reference]

    @property
    def bus(self) -> str:
        """Representative bus of the selected area."""
        return self.run.buses[self.area]

    def metrics(self) -> pd.DataFrame:
        """Area metrics of the run in the period."""
        return area_metrics(
            self.run.area_model.loc[self.hours], self.run.area_observed.loc[self.hours]
        )

    def bus_metrics(self) -> pd.DataFrame:
        """Bus metrics of the run in the period, as in the evaluation rule."""
        return bus_metrics(self.run.model, self.run.observed, self.hours)


def zone_label(area: str) -> str:
    """Plain name of an area with its code."""
    return f"{NAMES[area]} ({area})"


def takeaway(text: str) -> None:
    """Show the key finding of a page in a bordered box."""
    st.container(border=True).markdown(text)


def worst(values: pd.Series, n: int = 3, signed: bool = False) -> str:
    """List the areas with the largest absolute values and their values."""
    top = values.abs().nlargest(n).index if signed else values.nlargest(n).index
    template = "{:+.1f}" if signed else "{:.1f}"
    return ", ".join(f"{NAMES[a]} ({template.format(values[a])})" for a in top)


def export_direction(model: float, scheduled: float) -> str:
    """Whether the model exports too much or too little compared with the schedule."""
    return "too much" if model > scheduled else "too little"


def zone_height(rows: int) -> int:
    """Height of a per-area figure with one row per area."""
    return ROW_HEIGHT * rows + 110


def zone_layout(fig: go.Figure, areas: list[str]) -> go.Figure:
    """Order a horizontal per-area figure by zone with group separators and row-based height."""
    for y in group_breaks(areas):
        fig.add_hline(y=y, line_width=0.6, line_color="grey")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=zone_height(len(areas)), margin=dict(t=50, b=30))
    return fig


def region_choropleth(run: RunData, values: pd.Series, **style) -> go.Choropleth:
    """Choropleth of bus values over the bus regions, hovering the country name of each bus."""
    areas = run.area_to_bus[values.index]
    return go.Choropleth(
        geojson=run.regions,
        featureidkey="properties.name",
        locations=values.index,
        z=values.to_numpy(),
        customdata=areas.to_numpy(),
        text=[NAMES[area] for area in areas],
        hovertemplate="%{text}: %{z:.2f}<extra></extra>",
        **style,
    )


def map_layout(fig: go.Figure) -> go.Figure:
    """Fit the geo subplots of a region map to Europe without frame and margins."""
    fig.update_geos(**GEOS)
    fig.update_layout(height=MAP_HEIGHT, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def selected_points(event: dict | None) -> list[dict]:
    """Selected points of a Plotly selection event."""
    return [] if event is None else list(event["selection"]["points"])


def group_breaks(areas: list[str]) -> list[float]:
    """Positions between consecutive areas of different zone groups."""
    group = pd.Series({a: g for g, members in ZONE_GROUPS.items() for a in members})
    values = group[areas].to_numpy()
    return list(np.flatnonzero(values[1:] != values[:-1]) + 0.5)


def heatmap(
    data: pd.DataFrame,
    unit: str,
    diverging: bool = True,
    text: bool = False,
) -> go.Figure:
    """Draw an area heatmap with group separators and a transparent scatter layer to select cells."""
    rows, columns = [str(i) for i in data.index], [str(c) for c in data.columns]
    limit = symmetric_limit(data, 0.95)
    scale = (
        dict(colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit)
        if diverging
        else dict(colorscale=SEQUENTIAL.capitalize())
    )
    fig = go.Figure(
        go.Heatmap(
            z=data.to_numpy(),
            x=columns,
            y=rows,
            text=data.round(0).to_numpy() if text else None,
            texttemplate="%{text}" if text else None,
            hovertemplate="%{y} %{x}: %{z:.1f}<extra></extra>",
            colorbar=dict(title=unit, thickness=12),
            **scale,
        )
    )
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    fig.add_scatter(
        x=grid_x.ravel(),
        y=grid_y.ravel(),
        mode="markers",
        marker=dict(size=12, opacity=0.01, symbol="square"),
        customdata=np.stack([grid_y.ravel(), grid_x.ravel()], axis=1),
        hoverinfo="skip",
        showlegend=False,
    )
    if data.index.isin(ZONE_ORDER).all():
        for y in group_breaks(rows):
            fig.add_hline(y=y, line_width=1, line_color="grey")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    fig.update_layout(
        height=ROW_HEIGHT * len(rows) + 80,
        clickmode="event+select",
        margin=dict(t=10, b=30),
    )
    return fig


def price_traces(model: pd.Series, observed: pd.Series, name: str) -> list[go.Scatter]:
    """Model and observed price lines, hovering the error."""
    error = (model - observed).to_numpy()
    return [
        go.Scatter(
            x=observed.index,
            y=observed,
            name="market",
            line=dict(color=OBSERVED_COLOR, width=1),
        ),
        go.Scatter(
            x=model.index,
            y=model,
            name=name,
            line=dict(color=MODEL_COLOR, width=1),
            customdata=error,
            hovertemplate="%{y:.1f} (error %{customdata:.1f})",
        ),
    ]


def event_markers(model: pd.Series, shedding: pd.Series) -> list[go.Scatter]:
    """Markers on the model price line at negative-price and load-shedding hours."""
    masks = {
        "negative price": model < 0,
        "load shedding": shedding.reindex(model.index) > 0,
    }
    return [
        go.Scatter(
            x=model.index[mask],
            y=model[mask],
            mode="markers",
            name=name,
            marker=dict(color=MARKER_COLORS[name], size=5),
        )
        for name, mask in masks.items()
        if mask.any()
    ]


def show(fig: go.Figure, key: str, select: bool = False) -> list[dict]:
    """Render a figure and return the selected points if selection is enabled."""
    if not select:
        st.plotly_chart(fig, key=key, width="stretch")
        return []
    event = st.plotly_chart(
        fig, key=key, on_select="rerun", selection_mode="points", width="stretch"
    )
    return selected_points(event)
