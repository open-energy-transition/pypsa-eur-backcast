# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Audits the model fleet against the installed generation capacity of the ENTSO-E Transparency Platform.

The installed capacity per bidding zone area and production type is queried
for each year and cached as raw CSV per area and year. The production types
are mapped to model carriers with ``PRODUCTION_TYPE_CARRIERS`` and the model
carriers with ``MODEL_CARRIERS``, unmapped types are dropped. The model
capacity is the ``p_nom`` of the generators and storage units of a composed
network, summed over the buses of each area. Per year, the output holds the
model and ENTSO-E capacity per area and carrier, their gap (model minus
ENTSO-E) and the ENTSO-E production types behind each carrier. Years
without a composed network keep the ENTSO-E capacity and leave the model
capacity empty. Areas are those of the area-to-bus map valid for the year
(``data/entsoe_area_to_bus-pre2021.csv`` up to 2020), with the zones of IT and SE merged into their country, as ENTSO-E publishes no capacity for them. An API token must be set in the environment variable
``ENTSOE_API_TOKEN``. Run directly, as the audit is not part of the workflow.
"""

import argparse
import logging
from pathlib import Path

import netCDF4
import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from scripts.lib.entsoe import entsoe_client, read_area_to_bus, utc_year

logger = logging.getLogger(__name__)

PRODUCTION_TYPE_CARRIERS = {
    "Biomass": "biomass",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Oil shale": "lignite",
    "Fossil Peat": "lignite",
    "Fossil Coal-derived gas": "coal",
    "Fossil Hard coal": "coal",
    "Fossil Gas": "gas",
    "Fossil Oil": "oil",
    "Nuclear": "nuclear",
    "Solar": "solar",
    "Wind Onshore": "onwind",
    "Wind Offshore": "offwind",
    "Hydro Run-of-river and pondage": "ror",
    "Hydro Water Reservoir": "hydro",
    "Hydro Pumped Storage": "PHS",
    "Geothermal": "geothermal",
    "Waste": "waste",
}

MODEL_CARRIERS = {
    "CCGT": "gas",
    "OCGT": "gas",
    "offwind-ac": "offwind",
    "offwind-dc": "offwind",
    "offwind-float": "offwind",
    "solar-hsat": "solar",
}

COUNTRY_REPORTED_AREAS = ("IT", "SE")

COMPONENTS = ["generators", "storage_units"]
AREA_TO_BUS = "data/entsoe_area_to_bus.csv"
AREA_TO_BUS_PRE2021 = "data/entsoe_area_to_bus-pre2021.csv"
NETWORKS = {
    2019: "dev/day-ahead-backcast/converge/baseline/composed_2019.nc",
    2023: "dev/day-ahead-backcast/converge/baseline/composed_2023.nc",
    2024: "resources/backcast-2024/networks/composed_2024.nc",
}
GAP_THRESHOLD_MW = 500
FLEET_COLUMNS = ["area", "carrier", "model_mw", "entsoe_mw", "gap_mw", "entsoe_types"]


def query_installed_capacity(area: str, year: int, cache: Path) -> pd.DataFrame:
    """Return the installed capacity per production type of `area` and `year`, reading or filling the raw cache."""
    path = cache / f"installed_{area}_{year}.csv"
    if not path.exists():
        start, end = utc_year(year)
        try:
            raw = entsoe_client().query_installed_generation_capacity(
                area, start=start, end=end
            )
        except NoMatchingDataError:
            logger.warning(f"No installed capacity published for {area} in {year}.")
            raw = pd.DataFrame()
        raw.to_csv(path)
    return pd.read_csv(path, index_col=0)


def entsoe_capacity(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the first published capacity per area and model carrier in MW with the contributing production types."""
    rows = [
        (area, carrier, production_type, mw)
        for area, df in raw.items()
        if not df.empty
        for production_type, mw in df.iloc[0].dropna().items()
        if (carrier := PRODUCTION_TYPE_CARRIERS.get(production_type))
    ]
    frame = pd.DataFrame(
        rows, columns=["area", "carrier", "production_type", "entsoe_mw"]
    )
    frame["entsoe_types"] = (
        frame.production_type + " " + frame.entsoe_mw.round().astype(int).astype(str)
    )
    return frame.groupby(["area", "carrier"]).agg(
        entsoe_mw=("entsoe_mw", "sum"), entsoe_types=("entsoe_types", "; ".join)
    )


def reporting_area(area: str) -> str:
    """Return the area ENTSO-E publishes installed capacity for, the country for zones of IT and SE."""
    country = area.split("_")[0]
    return country if country in COUNTRY_REPORTED_AREAS else area


def read_p_nom(path: str) -> pd.DataFrame:
    """Read bus, carrier and p_nom of the generators and storage units without loading the time series."""
    with netCDF4.Dataset(path) as ds:
        return pd.concat(
            pd.DataFrame(
                {attr: ds[f"{c}_{attr}"][:] for attr in ["bus", "carrier", "p_nom"]}
            )
            for c in COMPONENTS
        )


def model_capacity(p_nom: pd.DataFrame, area_to_bus: pd.Series) -> pd.Series:
    """Sum p_nom in MW per area and model carrier."""
    areas = p_nom.bus.map(area_to_bus.to_dict())
    assert areas.notna().all(), (
        f"Buses without area: {sorted(set(p_nom.bus[areas.isna()]))}"
    )
    carriers = p_nom.carrier.replace(MODEL_CARRIERS)
    return (
        p_nom.p_nom.groupby([areas.rename("area"), carriers.rename("carrier")])
        .sum()
        .rename("model_mw")
    )


def fleet_audit(model: pd.Series | None, entsoe: pd.DataFrame) -> pd.DataFrame:
    """Join model and ENTSO-E capacity per area and carrier and add the gap, model minus ENTSO-E."""
    if model is None:
        fleet = entsoe.assign(model_mw=float("nan"))
    else:
        fleet = entsoe.join(model, how="outer").fillna({"entsoe_mw": 0, "model_mw": 0})
    fleet["gap_mw"] = fleet.model_mw - fleet.entsoe_mw
    return fleet.reset_index()[FLEET_COLUMNS].round(1)


def gap_table(fleets: dict[int, pd.DataFrame]) -> str:
    """Render the gaps above the threshold per year as markdown tables, largest first."""
    sections = []
    for year, fleet in fleets.items():
        if fleet.model_mw.isna().all():
            continue
        gaps = fleet[fleet.gap_mw.abs() > GAP_THRESHOLD_MW]
        gaps = gaps.reindex(gaps.gap_mw.abs().sort_values(ascending=False).index)
        sections.append(
            f"## {year}\n\n"
            + gaps.drop(columns="entsoe_types").to_markdown(index=False, floatfmt=".0f")
        )
    return "\n\n".join(sections) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    parser.add_argument("--out", default="dev/backcast-accuracy/fleet-audit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    out = Path(args.out)
    cache = out / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    fleets = {}
    for year in args.years:
        area_to_bus = read_area_to_bus(
            AREA_TO_BUS_PRE2021 if year <= 2020 else AREA_TO_BUS
        ).map(reporting_area)
        logger.info(f"Auditing the fleet of {year}...")
        raw = {
            area: query_installed_capacity(area, year, cache)
            for area in sorted(set(area_to_bus))
        }
        model = (
            model_capacity(read_p_nom(NETWORKS[year]), area_to_bus)
            if year in NETWORKS
            else None
        )
        fleets[year] = fleet_audit(model, entsoe_capacity(raw))
        fleets[year].to_csv(out / f"fleet_{year}.csv", index=False)

    (out / "gaps.md").write_text(gap_table(fleets))
