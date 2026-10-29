# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
The story: hero week, big numbers, accuracy per bidding zone, four puzzles, error patterns and a call to action.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import narrative
from dashboard.data import (
    EXPORT_MODEL,
    EXPORT_SCHEDULED,
    FOCUS_AREA,
    KPIS,
    TOP_ERROR,
    TOP_SHARE,
    RunData,
    hero_week,
    history_frame,
    kpis,
    puzzle_tables,
    summary,
)
from dashboard.state import follow_selection, open_page
from dashboard.views import (
    FLOWS_PAGE,
    METHOD_PAGE,
    ZONE_PAGE,
    energy,
    errors,
    replay,
)
from dashboard.views.common import (
    CHART_HEIGHT,
    Context,
    export_direction,
    map_layout,
    price_traces,
    region_choropleth,
    show,
    worst,
    zone_label,
    zone_layout,
)
from scripts.lib.backcast_data import price_metrics
from scripts.lib.backcast_plotting import (
    BAND_COLOR,
    DIVERGING,
    MODEL_COLOR,
    OBSERVED_COLOR,
    SEQUENTIAL,
    symmetric_limit,
)

PERCENT = {"MAE / mean"}
MAP_METRICS = {
    "MAE (€/MWh)": "mae",
    "bias (€/MWh)": "mean_bias",
    "Pearson r": "pearson_r",
    "std ratio": "std_ratio",
}
DEFAULT_MAP = "MAE (€/MWh)"
DIVERGING_METRICS = {"mean_bias"}
CARD_ZONES = 6
CARD_HEIGHT = 220
CARD_COLORS = {"market": OBSERVED_COLOR, "model": MODEL_COLOR}
TABLE_COLUMNS = {
    "mae": "MAE",
    "mean_bias": "bias",
    "rmse": "RMSE",
    "pearson_r": "r",
    "std_ratio": "std ratio",
    "mean_observed": "mean observed",
    "mean_model": "mean model",
    "nonpositive_observed": "hours ≤ 0 observed",
    "nonpositive_model": "hours ≤ 0 model",
}


@dataclass(frozen=True)
class Puzzle:
    """One puzzle card: plain question, answer filled from the run, chart data and the page its button opens."""

    question: str
    answer: str
    chart: pd.DataFrame
    unit: str
    action: str
    page: str
    area: str


def formatted(name: str, value: float) -> str:
    """Format a headline number."""
    return f"{value:.0%}" if name in PERCENT else f"{value:.2f}"


def kpi_row(label: str, metrics: pd.DataFrame) -> None:
    """Show one row of headline numbers."""
    values = kpis(metrics)
    columns = st.columns([1.4, *[1] * len(KPIS)])
    columns[0].markdown(f"**{label}**")
    for column, name in zip(columns[1:], KPIS):
        column.metric(name, formatted(name, values[name]))


def kpi_text(values: dict[str, float]) -> str:
    """Headline numbers as a sentence fragment."""
    return (
        f"MAE **{values['MAE']:.2f} €/MWh** "
        f"({values['MAE / mean']:.0%} of the mean price), "
        f"r **{values['r']:.2f}**, |bias| {values['|bias|']:.2f} €/MWh"
    )


def lead(ctx: Context, buses: pd.DataFrame, metrics: pd.DataFrame) -> str:
    """Summarise the quality of the run, the validation runs over the comparable period and the worst zones."""
    text = (
        f"Median over **{len(buses)} model regions** ({len(metrics)} bidding zones), "
        f"{ctx.period}: {kpi_text(kpis(buses))}."
    )
    for ref in ctx.references:
        text += f" Validation, {ref.period}: {kpi_text(kpis(ref.bus_metrics()))}."
    return (
        f"{text}\n\nLargest MAE: {worst(metrics.mae)} €/MWh. "
        f"Largest bias: {worst(metrics.mean_bias, signed=True)} €/MWh. "
        f"Hours at or below 0 €/MWh: observed {buses.nonpositive_observed.mean():.1%}, "
        f"model {buses.nonpositive_model.mean():.1%} (mean over model regions)."
    )


def zone_map(ctx: Context, column: str) -> go.Figure:
    """Choropleth of a bus metric over the bus regions, carrying the area of each bus and outlining the selected area."""
    values = price_metrics(
        ctx.run.model.loc[ctx.hours], ctx.run.observed.loc[ctx.hours]
    )[column]
    selected = ctx.run.area_to_bus[values.index].to_numpy() == ctx.area
    limit = symmetric_limit(values)
    scale = (
        dict(colorscale=DIVERGING, zmid=0, zmin=-limit, zmax=limit)
        if column in DIVERGING_METRICS
        else dict(colorscale=SEQUENTIAL.capitalize())
    )
    trace = region_choropleth(
        ctx.run,
        values,
        marker_line_width=np.where(selected, 2.0, 0.3),
        marker_line_color=np.where(selected, "black", "white"),
        colorbar=dict(thickness=12, len=0.8),
        **scale,
    )
    fig = map_layout(go.Figure(trace))
    fig.update_layout(clickmode="event+select")
    return fig


def bars(metrics: pd.DataFrame, area: str) -> go.Figure:
    """Dumbbell of mean prices and bars of MAE and Pearson r per area in zone order, the selected area in bold."""
    areas = list(metrics.index)
    fig = make_subplots(
        1,
        3,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=["mean price (€/MWh)", "MAE (€/MWh)", "Pearson r"],
    )
    for zone, row in metrics.iterrows():
        fig.add_scatter(
            x=[row.mean_observed, row.mean_model],
            y=[zone, zone],
            mode="lines",
            line=dict(color=BAND_COLOR),
            showlegend=False,
            hoverinfo="skip",
            row=1,
            col=1,
        )
    for column, color, name in [
        ("mean_observed", OBSERVED_COLOR, "market"),
        ("mean_model", MODEL_COLOR, "model"),
    ]:
        fig.add_scatter(
            x=metrics[column],
            y=areas,
            mode="markers",
            marker=dict(color=color),
            name=name,
            customdata=areas,
            row=1,
            col=1,
        )
    for col, column in [(2, "mae"), (3, "pearson_r")]:
        fig.add_bar(
            x=metrics[column],
            y=areas,
            orientation="h",
            marker_color=MODEL_COLOR,
            name="model",
            customdata=areas,
            showlegend=False,
            row=1,
            col=col,
        )
    fig.update_layout(
        clickmode="event+select",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    zone_layout(fig, areas)
    labels = [f"<b>{a}</b>" if a == area else a for a in areas]
    fig.update_yaxes(tickvals=areas, ticktext=labels)
    fig.update_layout(margin=dict(t=90))
    return fig


def hero_chart(ctx: Context, week: pd.DatetimeIndex) -> go.Figure:
    """Model and observed prices of the area in the week with the spike and the lowest hour at or below 0 €/MWh annotated."""
    model = ctx.run.area_model.loc[week, ctx.area]
    observed = ctx.run.area_observed.loc[week, ctx.area]
    fig = go.Figure(price_traces(model, observed, "model"))
    peak, low = observed.idxmax(), observed.idxmin()
    notes = {peak: ("spikes to", "reaches", -40)}
    if observed[low] <= 0:
        notes[low] = ("drops to", "stays at", 40)
    for hour, (market, response, shift) in notes.items():
        fig.add_annotation(
            x=hour,
            y=observed[hour],
            text=f"Market {market} {observed[hour]:.0f} €/MWh,<br>the model {response} {model[hour]:.0f}",
            ay=shift,
            showarrow=True,
            arrowhead=2,
        )
    fig.update_layout(
        height=CHART_HEIGHT,
        hovermode="x unified",
        yaxis_title="€/MWh",
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def hero(ctx: Context, week: pd.DatetimeIndex) -> None:
    """Draw the plain explanation and the hero week of the area."""
    st.markdown(narrative.HERO.format(year=ctx.run.year))
    st.plotly_chart(hero_chart(ctx, week), width="stretch")
    st.caption(narrative.HERO_CHART.format(area=zone_label(ctx.area), start=week[0]))


def big_numbers(ctx: Context) -> None:
    """Draw zones covered, shape match and typical hourly miss of the period with the validation runs as caption."""
    values = kpis(ctx.bus_metrics())
    zones, shape, gap = st.columns(3)
    zones.metric(
        "Bidding zones covered",
        len(ctx.metrics()),
        help="A bidding zone is a market area with one power price, mostly one country.",
    )
    shape.metric(
        "How well the hourly shape matches (r)",
        f"{values['r']:.2f}",
        help="Pearson r, median over the model regions (about one per bidding zone): 1 means the model follows every up and down of the market price, 0 means no match.",
    )
    gap.metric(
        "Typical hourly miss",
        f"{values['MAE']:.1f} €/MWh",
        help="Mean absolute error, median over the model regions: the average gap between model and market price.",
    )
    notes = [f"{ctx.run.name}, {ctx.period}."]
    for ref in ctx.references:
        checked = kpis(ref.bus_metrics())
        notes.append(
            f"Validation year {ref.run.year}, {ref.period}: shape match {checked['r']:.2f}, "
            f"typical miss {checked['MAE']:.1f} €/MWh."
        )
    st.caption(" ".join(notes))


def pick_country(ctx: Context) -> None:
    """Draw a zone selector with a button that opens the zone page."""
    st.header("Pick your country")
    areas = list(ctx.run.buses.index)
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    area = left.selectbox(
        narrative.PICK, areas, index=areas.index(ctx.area), format_func=zone_label
    )
    right.button(
        "Explore",
        key="pick_country",
        type="primary",
        width="stretch",
        on_click=open_page,
        args=(ZONE_PAGE,),
        kwargs=dict(area=area),
    )


def where_lead(metrics: pd.DataFrame, period: str) -> str:
    """Name the zones with the smallest and largest MAE."""
    best = ", ".join(
        f"{narrative.NAMES[a]} ({v:.1f})" for a, v in metrics.mae.nsmallest(3).items()
    )
    return narrative.WHERE.format(best=best, worst=worst(metrics.mae), period=period)


def accuracy(ctx: Context) -> None:
    """Draw the map and per-zone bars of the period with the definitions and metric table in expanders, a click on a zone opens it."""
    metrics = ctx.metrics()
    buses = ctx.bus_metrics()
    st.header("Where it works")
    st.markdown(where_lead(metrics, ctx.period))
    left, right = st.columns(2)
    with left:
        name = st.segmented_control(
            "Map metric", list(MAP_METRICS), default=DEFAULT_MAP, key="map_metric"
        )
        points = show(
            zone_map(ctx, MAP_METRICS[name or DEFAULT_MAP]), "zone_map", select=True
        )
    with right:
        points += show(bars(metrics, ctx.area), "overview_bars", select=True)
    area = points[0]["customdata"] if points else None
    follow_selection("overview", area, area=area, page=ZONE_PAGE)
    with st.expander("What the numbers mean"):
        st.markdown(narrative.DEFINITIONS)
        st.markdown(lead(ctx, buses, metrics))
        kpi_row(f"{ctx.run.name}, {ctx.period}", buses)
        for ref in ctx.references:
            kpi_row(f"validation {ref.run.name}, {ref.period}", ref.bus_metrics())
    table = metrics[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    shares = [c for c in table if "≤" in c]
    table[shares] *= 100
    with st.expander("Metric table per zone (€/MWh, shares in %)"):
        st.dataframe(table.style.format("{:.2f}"))


def plain_error(value: float) -> str:
    """A signed price error in plain words."""
    return f"{abs(value):.0f} €/MWh too {'low' if value < 0 else 'high'}"


def puzzles(run: RunData) -> list[Puzzle]:
    """The four puzzle cards of a run over its whole year."""
    tables = puzzle_tables(run.area_model, run.area_observed)
    spikes, negative, cheap = tables["spikes"], tables["negative"], tables["cheap"]
    spike_zone, cheap_zone = spikes.index[0], cheap.index[0]
    numbers = summary(run)
    export, scheduled = numbers[EXPORT_MODEL], numbers[EXPORT_SCHEDULED]
    shares = negative.loc[FOCUS_AREA]
    return [
        Puzzle(
            "Why are the price spikes too low?",
            narrative.SPIKES.format(
                top=f"{TOP_SHARE:.0%}",
                median=plain_error(numbers[TOP_ERROR]),
                zone=narrative.NAMES[spike_zone],
                value=plain_error(spikes.model.iloc[0]),
            ),
            spikes.head(CARD_ZONES),
            "€/MWh, model minus market",
            f"Explore {narrative.NAMES[spike_zone]}",
            ZONE_PAGE,
            spike_zone,
        ),
        Puzzle(
            "Why does the model almost never go negative?",
            narrative.NEGATIVE.format(
                zone=narrative.NAMES[FOCUS_AREA],
                market=f"{shares.market:.1%}",
                model=f"{shares.model:.1%}",
            ),
            100 * negative.head(CARD_ZONES),
            "% of hours at or below 0",
            f"Explore {narrative.NAMES[FOCUS_AREA]}",
            ZONE_PAGE,
            FOCUS_AREA,
        ),
        Puzzle(
            f"Why does {narrative.NAMES[FOCUS_AREA]} export {export_direction(export, scheduled)}?",
            narrative.EXPORT.format(
                zone=narrative.NAMES[FOCUS_AREA],
                model=f"{export:+.0f}",
                scheduled=f"{scheduled:+.0f}",
            ),
            pd.DataFrame({"market": scheduled, "model": export}, [FOCUS_AREA]),
            "net export (TWh)",
            f"See the trade of {narrative.NAMES[FOCUS_AREA]}",
            FLOWS_PAGE,
            FOCUS_AREA,
        ),
        Puzzle(
            f"Why is {narrative.NAMES[cheap_zone]} too cheap?",
            narrative.CHEAP.format(
                zone=narrative.NAMES[cheap_zone], value=plain_error(cheap.model.iloc[0])
            ),
            cheap.head(CARD_ZONES),
            "mean bias (€/MWh)",
            f"Explore {narrative.NAMES[cheap_zone]}",
            ZONE_PAGE,
            cheap_zone,
        ),
    ]


def card_chart(frame: pd.DataFrame, unit: str) -> go.Figure:
    """Small grouped bars of market and model per area."""
    fig = go.Figure(
        [
            go.Bar(x=frame.index, y=frame[c], name=c, marker_color=CARD_COLORS[c])
            for c in frame
        ]
    )
    fig.update_layout(
        barmode="group",
        height=CARD_HEIGHT,
        yaxis_title=unit,
        showlegend=len(frame.columns) > 1,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


def puzzle_card(puzzle: Puzzle, key: str) -> None:
    """Draw one puzzle card in a bordered box."""
    box = st.container(border=True)
    box.subheader(puzzle.question)
    box.markdown(puzzle.answer)
    box.plotly_chart(card_chart(puzzle.chart, puzzle.unit), key=f"{key}_chart")
    box.button(
        puzzle.action,
        key=key,
        on_click=open_page,
        args=(puzzle.page,),
        kwargs=dict(area=puzzle.area),
    )


def four_puzzles(run: RunData) -> None:
    """Draw the puzzle cards in a two by two grid."""
    st.header(narrative.PUZZLES)
    st.markdown(narrative.PUZZLES_LEAD.format(year=run.year))
    columns = [*st.columns(2), *st.columns(2)]
    for i, (column, puzzle) in enumerate(zip(columns, puzzles(run))):
        with column:
            puzzle_card(puzzle, f"puzzle_{i}")


def history_teaser() -> None:
    """Draw one line on how the model got better with a button to its history."""
    kept = history_frame(narrative.HISTORY).query("kept")["2023"].dropna()
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown(
        narrative.HISTORY_TEASER.format(start=kept.iloc[0], end=kept.iloc[-1])
    )
    right.button(
        "See every step",
        key="history_teaser",
        width="stretch",
        on_click=open_page,
        args=(METHOD_PAGE,),
    )


def call_to_action() -> None:
    """Draw the invitation to reuse and contribute with the project links and the history teaser."""
    st.header("Take it further")
    st.markdown(narrative.CTA)
    for column, (name, url) in zip(
        st.columns(len(narrative.LINKS)), narrative.LINKS.items()
    ):
        column.link_button(name, url, width="stretch")
    history_teaser()


def render(ctx: Context) -> None:
    """Draw the story page from hero to call to action."""
    week = hero_week(ctx.run.area_observed[ctx.area])
    hero(ctx, week)
    replay.replay(ctx, week)
    big_numbers(ctx)
    pick_country(ctx)
    accuracy(ctx)
    four_puzzles(ctx.run)
    energy.energy_mix(ctx)
    errors.error_patterns(ctx)
    call_to_action()
