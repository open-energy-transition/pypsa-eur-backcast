# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prose and recorded refinement history of the story and how-it-works pages, placeholders in braces are filled from
the selected run.
"""

from dataclasses import dataclass

TITLE = "Can costs explain Europe's power prices?"
CLAIM = "Fuel and CO2 costs explain most of Europe's power prices. The rest is the interesting part."

NAMES = {
    "DE_LU": "Germany",
    "FR": "France",
    "BE": "Belgium",
    "NL": "Netherlands",
    "AT": "Austria",
    "CH": "Switzerland",
    "CZ": "Czechia",
    "PL": "Poland",
    "SK": "Slovakia",
    "HU": "Hungary",
    "SI": "Slovenia",
    "HR": "Croatia",
    "RO": "Romania",
    "BG": "Bulgaria",
    "RS": "Serbia",
    "GR": "Greece",
    "ES": "Spain",
    "PT": "Portugal",
    "IT_NORD": "Italy North",
    "IT_CNOR": "Italy Centre-North",
    "IT_CSUD": "Italy Centre-South",
    "IT_SUD": "Italy South",
    "IT_CALA": "Italy Calabria",
    "IT_ROSN": "Italy Rossano",
    "IT_SICI": "Sicily",
    "IT_SARD": "Sardinia",
    "NO_1": "Norway South-East",
    "NO_2": "Norway South-West",
    "NO_3": "Norway Centre",
    "NO_4": "Norway North",
    "NO_5": "Norway West",
    "SE_1": "Sweden North",
    "SE_2": "Sweden North-Centre",
    "SE_3": "Sweden South-Centre",
    "SE_4": "Sweden South",
    "FI": "Finland",
    "DK_1": "Denmark West",
    "DK_2": "Denmark East",
    "EE": "Estonia",
    "LV": "Latvia",
    "LT": "Lithuania",
    "IE_SEM": "Ireland",
    "GB": "Great Britain",
    "AL": "Albania",
    "BA": "Bosnia and Herzegovina",
    "ME": "Montenegro",
    "MK": "North Macedonia",
    "XK": "Kosovo",
    "MD": "Moldova",
    "UA_IPS": "Ukraine",
}

HERO = """
We rebuilt Europe's power market of {year} in a computer model in which the cheapest plants run first.
That gives an hourly price for each bidding zone, a market
area with one price, mostly one country. We then compare it with the real day-ahead price.
"""

HERO_CHART = """
**{area}, week of {start:%d %B %Y}.** Picked from the data: the week with the widest swing in market prices
that also has an hour at or below zero, if there is one.
"""

PICK = "Pick a country to see its prices, power plants and trade hour by hour."

WHERE = """
The model comes closest in {best} and misses most in {worst} (typical hourly miss in €/MWh, {period}).
Click a zone on the map or a bar to explore it.
"""

DEFINITIONS = """
- **Typical hourly miss (MAE, mean absolute error)**: the average gap between model and market price, in €/MWh.
- **Shape match (Pearson r)**: how well the model follows the ups and downs of the market price. 1 is perfect, 0 is no match.
- **Bias**: the model's yearly mean price minus the market's. Positive means the model is too expensive.
- **std ratio**: how strongly the model price swings compared with the market. 1 means equally strong.
- Headline numbers are medians over the model regions, about one per bidding zone, as in the evaluation of a run.
"""

PUZZLES = "Four puzzles"
PUZZLES_LEAD = """
Where the model misses, it teaches us something about the market. These are the four biggest gaps.
They cover the full year {year}, whatever period is picked in the sidebar.
"""

SPIKES = """
In the {top} most expensive hours of the year the model price is {median} (median over model regions). In {zone} it is {value}.
Real bids carry a markup on top of fuel and CO2: scarcity rents and start-up costs. A cost-only model cannot see them.
"""

NEGATIVE = """
In {zone} the market price was at or below zero in {market} of hours over the year. The model gets there in {model}.
In the model, nuclear, biomass and Nordic hydro power down at a small positive cost and soak up every surplus.
Real plants are less flexible.
"""

EXPORT = """
Net export of {zone} over the year, that is sales abroad minus purchases: {model} TWh in the model,
{scheduled} TWh scheduled in the market.
The model trades freely up to the agreed border capacity. It has no physical grid, so limits inside the grid
that shape real trade are missing, and power flows wherever the price gap is largest.
"""

CHEAP = """
The price of {zone} is {value} on average over the year. Two common causes. Cheap imports from
neighbours with flat prices can pull the price down. And the water flowing into hydro plants, and its split between
zones of one country, is uncertain.
"""

ENERGY = """
Prices follow the plants that run. Below, the energy of each kind of plant in the model is set against
what ENTSO-E reports, first summed over all zones, then per zone.
"""

ENERGY_LEAD = """
Across the {zones} zones with ENTSO-E generation data the model produces **{model:.0f} TWh**, ENTSO-E reports
**{entsoe:.0f} TWh**. Largest gaps per kind of plant: {gaps}. Largest single gap: **{carrier}** in {zone}, {value:+.1f} TWh.
"""

MISSES = """
The misses are not random. Below they are split by month, by price level and by hour of day.
Red means the model is too expensive, blue means too cheap.
"""

CTA = """
Everything here is open: the model, the data pipeline and this dashboard. Reuse it, question it, and help improve it.
"""

LINKS = {
    "Backcast on GitHub": "https://github.com/open-energy-transition/pypsa-eur-backcast",
    "PyPSA-Eur": "https://github.com/PyPSA/pypsa-eur",
    "Open Energy Transition": "https://openenergytransition.org",
}

WHAT = """
The backcast replays a past year hour by hour. It asks one question: if every power plant offered its power at
its fuel and CO2 cost, which day-ahead price would each bidding zone have seen? It compares that price with the
real one, as published on ENTSO-E, the data platform of Europe's grid operators. The validation year, if cached,
checks that improvements are not tuned to one year.
"""

HOW = """
The model is PyPSA-Eur, an open model of the European power system. Here it only decides which plants run in each
hour: all plants, lines and storage are fixed to what existed in that year. A solver plans the whole year at once,
knowing all hours in advance. Plants of one type do not all cost the same, so their cost rises with output. The
model price in a zone is the cost of serving one more MWh there.
"""

PIPELINE = {
    "Inputs": [
        "real electricity demand per zone from ENTSO-E",
        "wind and solar from weather data, wind and nuclear matched to ENTSO-E output",
        "the power plants of the year, with a cost range per plant type",
        "monthly fuel and CO2 prices",
        "hourly trading capacity between zones",
        "water flowing into hydro plants, pumped storage sized per country",
    ],
    "Model": [
        "about one region per bidding zone",
        "zones trade up to the agreed border capacity, no physical grid",
        "cheapest dispatch for every hour of the year",
        "one optimisation for the whole year, knowing all hours in advance",
    ],
    "Output": [
        "hourly price per zone",
        "power plant output per type and trade between zones",
        "scored against ENTSO-E day-ahead prices",
    ],
}

SCORING = """
For each zone the model price is compared with the real hourly price over the full year. Three numbers
summarise a run, each as the median over the model regions (about one per bidding zone):

- **MAE** (mean absolute error): the typical hourly miss in €/MWh.
- **Pearson r**: how well the hourly ups and downs follow the market. 1 is perfect.
- **|bias|**: how far the yearly mean price is off, in €/MWh.

A change to the model is kept only if it lowers the median MAE in both the main and the validation year.
This guards against tuning to one year. The one exception is a data fix to the zone shapes, kept despite a small
loss in 2023. Zones that publish prices in a local currency in a year are not scored.
"""

WORKS = """
- The hourly ups and downs follow the market in most zones: the median r is {r} in {run}.
- In tight hours the model runs the same plants as the market. The gap is in the price, not in which plants run.
- Most changes that were kept fix input data (wind output, plant data, zone shapes, pumped storage size).
  They are not tuning knobs, and they pass in both years.
"""

LIMITATIONS = """
| What is missing | Evidence in {run} | Where to look |
|---|---|---|
| Price spikes are too low. Real bids carry markups (scarcity rents, start-up costs) that a cost-only model cannot produce. | mean miss in the {top} most expensive hours: {top_error} €/MWh (median over model regions) | The story, four puzzles |
| Almost no hours at or below 0 €/MWh. Nuclear, biomass and Nordic hydro power down at a positive cost and soak up every surplus. Limits on how flexible plants are, are missing. | hours ≤ 0: {nonpositive_observed} in the market vs {nonpositive_model} in the model (mean over model regions) | The story, metric table |
| {exporter} exports {direction}. | net export {export_model} TWh in the model vs {export_scheduled} TWh scheduled | Trade between countries, {exporter} |
| Some zones are off in level, too expensive or too cheap on average. | too expensive by: {high_bias}; too cheap by: {low_bias} €/MWh | The story, map metric bias |
| Some zones miss the daily shape. Cheap imports from neighbours with flat prices can flatten the profile. | largest MAE: {worst_mae} | Explore a country |
| Without a physical grid, neighbouring zones can share one price more often than in the market. | | Trade between countries |
| Water flowing into hydro plants, and its split between zones of one country (SE, NO, IT), is uncertain. Reservoirs do not price their water from real filling levels. | | Explore a country, energy per type |
| Plant data has gaps; Dutch gas plants needed local fixes. Gas prices are monthly, not daily. Plants start and ramp without limits. | | Explore a country, energy per type |
"""

HISTORY_LEAD = """
Every change to the model was tested in both years. It was kept only if the typical hourly miss (median MAE over
model regions) went down. The one exception is a data fix, kept despite a small loss in 2023. Filled dots are kept
changes, hollow dots rejected ones. The line follows the kept model. Hover a dot to see the change. Numbers are as
recorded when each step was tested.
"""

HISTORY_TEASER = """
The model got here step by step. Each change was tested in two years, and the typical hourly miss in 2023 fell
from {start:.1f} to {end:.1f} €/MWh.
"""

REPLAY = "Play this week on the map: market and model prices hour by hour"
REPLAY_LEAD = """
Each region is one price region of the model, about one per bidding zone. Press play to watch prices move through the week in the market (left) and in
the model (right). The same colours mean the same price on both maps.
"""


@dataclass(frozen=True)
class Step:
    """One recorded refinement step with its median MAE in €/MWh per year, or a note when no MAE was recorded."""

    name: str
    verdict: str
    kept: bool
    mae_2023: float | None = None
    mae_2019: float | None = None
    note: str = ""


HISTORY = [
    Step("Baseline with NTC links and nominal prices", "start", True, 28.5, 9.9),
    Step(
        "Wind and nuclear calibrated to ENTSO-E, lignite fuel, flat hydro dispatch",
        "kept",
        True,
        24.0,
    ),
    Step(
        "Must-run floors, thermal cost spread, hybrid Core capacities",
        "kept",
        True,
        22.4,
        8.6,
    ),
    Step(
        "Iberian gas price cap, GB wind targets, wider cost spread, fixed external borders",
        "kept",
        True,
        21.8,
        8.2,
    ),
    Step(
        "Bidding-zone shapes from zone polygons",
        "kept as a data fix despite the 2023 loss",
        True,
        22.3,
        8.1,
    ),
    Step("Pumped storage energy capped at Geth et al. 2015", "kept", True, 21.2, 8.0),
    Step(
        "Dutch CCGT data fix and cost spread referenced to observed output",
        "kept",
        True,
        21.0,
        7.9,
    ),
    Step(
        "Biomass output capped at ENTSO-E capacity factors",
        "rejected",
        False,
        24.1,
        9.1,
    ),
    Step("Solar calibrated to ENTSO-E", "rejected, worse in 2019", False, 22.1, 8.2),
    Step(
        "Monthly thermal availability from observed peaks",
        "rejected, load shedding in 2019",
        False,
        22.1,
        10.8,
    ),
    Step(
        "Cost spread with fixed utilisation", "rejected", False, note="level +7 to +10"
    ),
    Step(
        "Annual hydro inflow scaling per zone",
        "rejected, r and mean MAE worse",
        False,
        note="median gain only",
    ),
    Step(
        "Reservoir targets, unit commitment, negative renewable bids, reserve demand curves",
        "rejected",
        False,
        note="no gain or too slow",
    ),
]
