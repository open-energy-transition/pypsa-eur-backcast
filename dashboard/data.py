# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Loaders of the dashboard extracts and pure derived frames.
"""

import calendar
import json
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dashboard.narrative import Step
from scripts.lib.backcast_data import (
    decile_errors,
    monthly_mae,
    price_metrics,
    top_errors,
)
from scripts.lib.backcast_plotting import (
    area_prices,
    representative_buses,
    select_weeks,
)

ROOT = Path(__file__).parent.parent
CACHE = Path(__file__).parent / "cache"
DATA_URL = "https://github.com/open-energy-transition/pypsa-eur-backcast/releases/download/dashboard-data/dashboard-cache.tar.gz"
FETCH_MARKER = "SOURCE"
MONTH_NAMES = list(calendar.month_name)[1:]
PERIODS = ["year", *MONTH_NAMES, "winter week", "summer week", "custom"]
AGGREGATIONS = {"hourly": None, "daily": "D"}
TOP_SHARE = 0.05
FOCUS_AREA = "DE_LU"
GENERATION_SOURCES = ["model", "entsoe"]
TOP_ERROR = f"error in top {TOP_SHARE:.0%} hours (€/MWh)"
EXPORT_MODEL = f"{FOCUS_AREA} net export model (TWh)"
EXPORT_SCHEDULED = f"{FOCUS_AREA} net export scheduled (TWh)"
WEEK_HOURS = 168
COORDINATE_DIGITS = 2
KPIS = {
    "MAE": ("mae", "median"),
    "MAE / mean": ("rel_mae", "median"),
    "r": ("pearson_r", "median"),
    "|bias|": ("abs_bias", "median"),
}


@dataclass(frozen=True)
class RunData:
    """Extracted prices, dispatch and flows of one solved backcast run."""

    name: str
    model: pd.DataFrame
    observed: pd.DataFrame
    load: pd.DataFrame
    generation: pd.DataFrame
    flows: pd.DataFrame
    marginal: pd.DataFrame
    metrics: pd.DataFrame
    monthly_bias: pd.DataFrame
    regions: dict
    area_to_bus: pd.Series
    manifest: dict
    buses: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        buses = representative_buses(self.area_to_bus, self.observed.columns)
        object.__setattr__(self, "buses", buses)

    @property
    def year(self) -> int:
        """Year of the backcast."""
        return self.manifest["year"]

    @property
    def area_model(self) -> pd.DataFrame:
        """Hourly model prices per area."""
        return area_prices(self.model, self.buses)

    @property
    def area_observed(self) -> pd.DataFrame:
        """Hourly observed prices per area."""
        return area_prices(self.observed, self.buses)

    @property
    def weeks(self) -> dict[str, pd.DatetimeIndex]:
        """Winter and summer week selected by the observed prices of the first area."""
        return select_weeks(self.area_observed.iloc[:, 0])


def load_run(path: Path) -> RunData:
    """Read the extract of a run."""
    frames = {
        name: pd.read_parquet(path / f"{name}.parquet")
        for name in ["prices", "load", "generation", "flows", "marginal"]
    }
    frames["metrics"] = pd.read_parquet(path / "metrics.parquet")
    monthly = pd.read_parquet(path / "monthly_bias.parquet")
    frames["monthly_bias"] = monthly.set_axis(pd.PeriodIndex(monthly.index, freq="M"))
    prices = frames.pop("prices")
    return RunData(
        name=path.name,
        model=prices["model"],
        observed=prices["observed"],
        regions=json.loads(
            (path / "regions.geojson").read_text(),
            parse_float=lambda x: round(float(x), COORDINATE_DIGITS),
        ),
        area_to_bus=pd.read_csv(path / "area_to_bus.csv", index_col="bus").iloc[:, 0],
        manifest=json.loads((path / "manifest.json").read_text()),
        **frames,
    )


def discover_runs(cache: Path) -> list[str]:
    """List the runs with an extract in the cache."""
    return sorted(p.parent.name for p in cache.glob("*/manifest.json"))


def fetch_cache(cache: Path, url: str = DATA_URL) -> None:
    """Download the packed extracts, unpack them into the cache and mark it as fetched."""
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            archive.extractall(cache, filter="data")
    (cache / FETCH_MARKER).write_text(url)


def stale_sources(manifest: dict) -> list[str]:
    """List sources modified after the extract or removed since."""
    return [
        source
        for source, mtime in manifest["sources"].items()
        if not Path(ROOT / source).exists()
        or Path(ROOT / source).stat().st_mtime > mtime
    ]


def period_index(
    index: pd.DatetimeIndex,
    preset: str,
    weeks: dict[str, pd.DatetimeIndex],
    dates: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """Select the hours of a period preset: the year, a month by name, a week or custom dates."""
    if preset == "year":
        return index
    if preset in MONTH_NAMES:
        return index[index.month == MONTH_NAMES.index(preset) + 1]
    if preset in ("winter week", "summer week"):
        return index.intersection(weeks[preset.removesuffix(" week")])
    assert preset == "custom" and dates is not None, f"Unknown preset {preset}"
    start, end = dates
    return index[(index >= start) & (index < end + pd.Timedelta(days=1))]


def aggregate(frame: pd.DataFrame | pd.Series, aggregation: str) -> pd.DataFrame:
    """Average to the aggregation frequency."""
    freq = AGGREGATIONS[aggregation]
    return frame if freq is None else frame.resample(freq).mean()


def area_metrics(model: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Compute the price metrics per area with absolute bias, relative MAE and shares of hours at or below zero."""
    metrics = price_metrics(model, observed).rename_axis("area")
    return metrics.assign(
        abs_bias=metrics.mean_bias.abs(),
        rel_mae=metrics.mae / metrics.mean_observed,
        nonpositive_model=metrics.zero_share_model + metrics.negative_share_model,
        nonpositive_observed=metrics.zero_share_observed
        + metrics.negative_share_observed,
    )


def bus_metrics(
    model: pd.DataFrame, observed: pd.DataFrame, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Price metrics per bus in the hours."""
    return area_metrics(model.loc[hours], observed.loc[hours]).rename_axis("bus")


def kpis(metrics: pd.DataFrame) -> dict[str, float]:
    """Summarise area metrics to the headline numbers."""
    return {name: metrics[col].agg(how) for name, (col, how) in KPIS.items()}


def month_errors(error: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Mean bias and MAE per area and month."""
    months = error.index.to_period("M")
    return {"bias": error.groupby(months).mean().T, "MAE": monthly_mae(error).T}


def level_errors(error: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Mean error per area and observed price decile with the top share of hours."""
    top = top_errors(error, observed, TOP_SHARE).rename(f"top {TOP_SHARE:.0%}")
    return decile_errors(error, observed).join(top)


def month_hour_errors(error: pd.Series, hours: pd.Index) -> pd.DataFrame:
    """Mean error per month and hour of day."""
    return error.groupby([error.index.month, hours]).mean().unstack()


def worst_hours(profile: pd.Series) -> tuple[int, int, float]:
    """Contiguous hours around the largest mean error sharing its sign and half its size, with their mean."""
    peak = profile.abs().idxmax()
    strong = (profile * profile[peak] > 0) & (profile.abs() >= profile.abs()[peak] / 2)
    first = last = peak
    while first - 1 in strong.index and strong[first - 1]:
        first -= 1
    while last + 1 in strong.index and strong[last + 1]:
        last += 1
    return first, last, profile.loc[first:last].mean()


def tail_shares(prices: pd.Series, levels: tuple[float, float] = (0.0, 150.0)) -> dict:
    """Shares of hours at or below the lower and above the upper price level."""
    low, high = levels
    return {
        f"≤ {low:.0f}": (prices <= low).mean(),
        f"> {high:.0f}": (prices > high).mean(),
    }


def source_generation(generation: pd.DataFrame, source: str, area: str) -> pd.DataFrame:
    """Hourly generation per carrier of an area and source, empty if the source lacks the area."""
    frame = generation[source]
    if area not in frame.columns.get_level_values("area"):
        return pd.DataFrame(index=frame.index)
    return frame[area]


def energy_errors(energy: dict[str, pd.Series]) -> pd.DataFrame:
    """Model and ENTSO-E energy per carrier with absolute and relative error, without carriers that are zero in both."""
    table = pd.DataFrame(energy).fillna(0.0)
    table = table[(table != 0).any(axis=1)]
    table["error"] = table.model - table.entsoe
    table["error %"] = 100 * table.error / table.entsoe.where(table.entsoe > 0)
    return table


def carrier_errors(
    generation: pd.DataFrame, area: str, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Model and ENTSO-E energy in TWh per carrier of an area with absolute and relative error."""
    return energy_errors(
        {
            source: source_generation(generation, source, area).reindex(hours).sum()
            / 1e6
            for source in GENERATION_SOURCES
        }
    )


def generation_areas(generation: pd.DataFrame) -> pd.Index:
    """Areas with generation in both the model and ENTSO-E."""
    model, entsoe = (generation[s].columns.unique("area") for s in GENERATION_SOURCES)
    return model.intersection(entsoe)


def area_energy(
    generation: pd.DataFrame, source: str, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Energy in TWh per area and carrier of a source over the hours, for the areas with generation in both sources."""
    frame = generation[source].reindex(hours)
    frame = frame.loc[
        :, frame.columns.get_level_values("area").isin(generation_areas(generation))
    ]
    return frame.sum().unstack("carrier").fillna(0.0) / 1e6


def total_carrier_errors(
    generation: pd.DataFrame, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Model and ENTSO-E energy in TWh per carrier summed over the areas with generation in both sources."""
    return energy_errors(
        {
            source: area_energy(generation, source, hours).sum()
            for source in GENERATION_SOURCES
        }
    )


def border_positions(
    flows: pd.DataFrame, area: str, hours: pd.DatetimeIndex
) -> pd.DataFrame:
    """Hourly export of an area per border for model and scheduled with the NTC in both directions."""
    columns = {}
    for border in flows["model"].columns:
        a, b = border.split("->")
        if area not in (a, b):
            continue
        sign, other = (1.0, b) if a == area else (-1.0, a)
        for source in ["model", "scheduled"]:
            if border in flows[source]:
                columns[(other, source)] = sign * flows[source][border]
        for key, direction in [
            ("ntc out", f"{area}->{other}"),
            ("ntc in", f"{other}->{area}"),
        ]:
            if direction in flows["ntc"]:
                columns[(other, key)] = flows["ntc"][direction]
    frame = pd.DataFrame(columns).reindex(hours)
    frame.columns.names = ["neighbour", "series"]
    return frame


def net_export(flows: pd.DataFrame, area: str, hours: pd.DatetimeIndex) -> pd.Series:
    """Net export of an area in TWh over the hours, modelled and scheduled."""
    positions = border_positions(flows, area, hours)
    series = positions.T.groupby(level="series").sum().T.sum() / 1e6
    return series.reindex(["model", "scheduled"], fill_value=0.0)


def summary(run: RunData) -> dict[str, float]:
    """Headline numbers of a run over its whole year, medians over model regions."""
    buses = bus_metrics(run.model, run.observed, run.observed.index)
    error = run.model - run.observed
    export = net_export(run.flows, FOCUS_AREA, run.observed.index)
    return {
        "year": run.year,
        "scored model regions": len(buses),
        "mean observed price (€/MWh)": buses.mean_observed.mean(),
        "median MAE (€/MWh)": buses.mae.median(),
        "median Pearson r": buses.pearson_r.median(),
        "median |bias| (€/MWh)": buses.abs_bias.median(),
        "hours ≤ 0 observed (%)": 100 * buses.nonpositive_observed.mean(),
        "hours ≤ 0 model (%)": 100 * buses.nonpositive_model.mean(),
        TOP_ERROR: top_errors(error, run.observed, TOP_SHARE).median(),
        EXPORT_MODEL: export["model"],
        EXPORT_SCHEDULED: export["scheduled"],
    }


def hero_week(prices: pd.Series) -> pd.DatetimeIndex:
    """Full Monday-start week with the widest price range among weeks with an hour at or below 0 €/MWh, else among all full weeks."""
    start = prices.index.normalize() - pd.to_timedelta(prices.index.weekday, unit="D")
    weeks = prices.groupby(start).agg(["count", "min", "max"])
    weeks = weeks[weeks["count"] == WEEK_HOURS]
    span = weeks["max"] - weeks["min"]
    negative = span[weeks["min"] <= 0]
    week = (negative if len(negative) else span).idxmax()
    return pd.date_range(week, periods=WEEK_HOURS, freq="h")


def puzzle_tables(
    model: pd.DataFrame, observed: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Areas worst first by error in the top share of expensive hours, by hours at or below 0 €/MWh in the market and by mean bias."""
    metrics = area_metrics(model, observed)
    top = top_errors(model - observed, observed, TOP_SHARE)
    shares = metrics[["nonpositive_observed", "nonpositive_model"]]
    return {
        "spikes": top.sort_values().to_frame("model"),
        "negative": shares.set_axis(["market", "model"], axis=1).sort_values(
            "market", ascending=False
        ),
        "cheap": metrics.mean_bias.sort_values().to_frame("model"),
    }


def marginal_shares(marginal: pd.Series) -> pd.Series:
    """Share of hours per price-setting carrier."""
    return marginal.astype(str).value_counts(normalize=True).rename("share")


def history_result(step: Step) -> str:
    """Recorded median MAE of a step as 2023 / 2019, or its note."""
    if step.mae_2023 is None:
        return step.note
    second = "-" if step.mae_2019 is None else f"{step.mae_2019:.1f}"
    return f"{step.mae_2023:.1f} / {second}"


def history_frame(steps: list[Step]) -> pd.DataFrame:
    """Refinement steps numbered from 1 with their median MAE per year, whether they were kept and the result as text."""
    return pd.DataFrame(
        {
            "step": [s.name for s in steps],
            "2023": [s.mae_2023 for s in steps],
            "2019": [s.mae_2019 for s in steps],
            "kept": [s.kept for s in steps],
            "result": [history_result(s) for s in steps],
            "verdict": [s.verdict for s in steps],
        },
        index=pd.RangeIndex(1, len(steps) + 1, name="#"),
    )
