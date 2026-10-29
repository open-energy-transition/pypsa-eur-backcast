# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Energy mix section of the story: model and ENTSO-E generation per carrier summed over all bidding zones and its error per zone.
"""

import pandas as pd
import streamlit as st

from dashboard import narrative
from dashboard.data import GENERATION_SOURCES, area_energy, total_carrier_errors
from dashboard.views.common import Context, heatmap, show, takeaway
from dashboard.views.dispatch import carrier_bars
from dashboard.views.errors import follow_zone
from scripts.lib.backcast_plotting import STACK, ZONE_ORDER

GAPS = 3
CARRIER_LABELS_MARGIN = 80


def stack_order(carriers: pd.Index) -> list[str]:
    """Carriers in the order of the generation stack, unknown carriers last."""
    return sorted(carriers, key=lambda c: STACK.index(c) if c in STACK else len(STACK))


def zone_errors(ctx: Context) -> pd.DataFrame:
    """Energy error model minus ENTSO-E in TWh per area and carrier, ordered by zone and stack."""
    model, entsoe = (
        area_energy(ctx.run.generation, source, ctx.hours)
        for source in GENERATION_SOURCES
    )
    error = model.sub(entsoe, fill_value=0.0)
    error = error.loc[:, (error != 0).any()]
    areas = sorted(
        error.index,
        key=lambda a: ZONE_ORDER.index(a) if a in ZONE_ORDER else len(ZONE_ORDER),
    )
    return error.loc[areas, stack_order(error.columns)]


def lead(table: pd.DataFrame, errors: pd.DataFrame) -> str:
    """Summarise the total energy, the largest carrier gaps and the largest gap of one zone and carrier."""
    gaps = table.error.abs().nlargest(GAPS).index
    named = ", ".join(f"**{c}** {table.error[c]:+.0f} TWh" for c in gaps)
    cells = errors.stack()
    area, carrier = cells.abs().idxmax()
    return narrative.ENERGY_LEAD.format(
        zones=len(errors),
        model=table.model.sum(),
        entsoe=table.entsoe.sum(),
        gaps=named,
        zone=narrative.NAMES[area],
        carrier=carrier,
        value=cells[area, carrier],
    )


def energy_mix(ctx: Context) -> None:
    """Draw the energy per carrier over all zones and its error per zone, a click on a row opens the zone."""
    table = total_carrier_errors(ctx.run.generation, ctx.hours)
    table = table.loc[stack_order(table.index)]
    errors = zone_errors(ctx)
    st.header("Do the right power plants run?")
    st.markdown(narrative.ENERGY)
    takeaway(lead(table, errors))
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Energy per carrier, all zones")
        show(carrier_bars(table), "energy_total")
        st.caption(f"TWh, {ctx.period}, zones with ENTSO-E generation data.")
    with right:
        st.subheader("Gap per zone")
        figure = heatmap(errors, "TWh")
        figure.update_layout(margin=dict(b=CARRIER_LABELS_MARGIN))
        follow_zone("energy_zone", show(figure, "energy_zone", select=True))
        st.caption(
            f"Model minus ENTSO-E, TWh, {ctx.period}. Click a row to open the zone."
        )
