# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Readers for the Eurostat gross production of electricity and heat by plant type and fuel (``nrg_bal_peh``).
"""

import pandas as pd

EUROSTAT_COUNTRIES = {"EL": "GR", "UK": "GB"}
GROSS_ELECTRICITY_SIEC = {"biomass": {"BIOE": 1.0, "W6210": -1.0}}


def read_nrg_bal_peh(path: str, year: int) -> pd.DataFrame:
    """Read the ``nrg_bal_peh`` values in MWh of `year` with the balance, fuel and country of each row."""
    raw = pd.read_csv(path, sep="\t", na_values=":")
    keys = raw.iloc[:, 0].str.split(",", expand=True)
    keys.columns = ["freq", "nrg_bal", "siec", "unit", "geo"]
    columns = {c.strip(): c for c in raw.columns}
    assert str(year) in columns, f"Eurostat nrg_bal_peh has no data for {year}."
    values = pd.to_numeric(
        raw[columns[str(year)]].astype(str).str.extract(r"([-\d.]+)")[0]
    )
    df = keys.assign(value=values.mul(1e3)).query("unit == 'GWH'")
    df["country"] = df.geo.replace(EUROSTAT_COUNTRIES)
    return df[["nrg_bal", "siec", "country", "value"]]


def gross_electricity(path: str, year: int, carrier: str) -> pd.Series:
    """Return the gross electricity production in MWh of `carrier` per country in `year`, all producers including autoproducers."""
    weights = pd.Series(GROSS_ELECTRICITY_SIEC[carrier])
    df = read_nrg_bal_peh(path, year).query(
        "nrg_bal == 'GEP' and siec in @weights.index"
    )
    table = df.pivot(index="country", columns="siec", values="value")
    table = table.reindex(columns=weights.index)
    return table.fillna(0).dot(weights)[table["BIOE"].notna()]


def eurostat_year(snapshots: pd.DatetimeIndex, year: int | None) -> int:
    """Return the Eurostat reference year, ``year`` if set, else the single snapshot year."""
    years = snapshots.year.unique()
    assert len(years) == 1, (
        f"Eurostat data need snapshots in one year, got {list(years)}."
    )
    return int(years[0]) if year is None else year
