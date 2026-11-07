# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for retrieve_day_ahead_prices_entsoe.py."""

import pandas as pd
import pytest
from entsoe import EntsoeRawClient

from scripts.retrieve_day_ahead_prices_entsoe import (
    currency_per_timestamp,
    parse_document_currencies,
    query_document_currencies,
)


def document(*timeseries: tuple[str, str]) -> str:
    series = "".join(
        f"<TimeSeries><currency_Unit.name>{currency}</currency_Unit.name>"
        f"<Period><timeInterval><start>{start}</start><end>x</end></timeInterval>"
        "</Period></TimeSeries>"
        for currency, start in timeseries
    )
    return (
        "<Publication_MarketDocument><period.timeInterval><start>2021-12-31T23:00Z"
        f"</start></period.timeInterval>{series}</Publication_MarketDocument>"
    )


def test_parse_document_currencies():
    xml = document(("BGN", "2022-01-20T23:00Z"), ("EUR", "2022-01-21T23:00Z"))

    result = parse_document_currencies(xml)

    assert result.tolist() == ["BGN", "EUR"]
    assert result.index.tolist() == [
        pd.Timestamp("2022-01-20 23:00", tz="UTC"),
        pd.Timestamp("2022-01-21 23:00", tz="UTC"),
    ]


@pytest.mark.parametrize(
    "timeseries, expected",
    [
        ([("BGN", "2022-01-01T00:00Z"), ("BGN", "2022-01-01T00:00Z")], ["BGN", "EUR"]),
        (
            [("BGN", "2022-01-01T00:00Z"), ("EUR", "2022-02-01T00:00Z")],
            ["BGN", "EUR", "EUR"],
        ),
        ([("BGN", "2022-01-01T00:00Z"), ("EUR", "2022-01-01T00:00Z")], None),
    ],
    ids=["duplicate", "switch", "conflict"],
)
def test_query_document_currencies(monkeypatch, timeseries, expected):
    def query(client, area, start, end):
        return document(
            *[(c, s) for c, s in published if start <= pd.Timestamp(s) < end]
        )

    published = [*timeseries, ("EUR", "2022-02-02T00:00Z")]

    monkeypatch.setattr(EntsoeRawClient, "query_day_ahead_prices", query)
    start, end = (
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-03-01", tz="UTC"),
    )

    if expected is None:
        with pytest.raises(AssertionError, match="two currencies"):
            query_document_currencies(None, "BG", start, end)
    else:
        result = query_document_currencies(None, "BG", start, end)
        assert result.tolist() == expected


@pytest.mark.parametrize(
    "first, expected",
    [
        ("2022-01-20 22:00", ["BGN", "BGN", "EUR", "EUR"]),
        ("2022-01-20 21:00", None),
    ],
    ids=["switch", "before-first-document"],
)
def test_currency_per_timestamp(first, expected):
    currencies = pd.Series(
        ["BGN", "EUR"],
        pd.DatetimeIndex(["2022-01-20 22:00", "2022-01-21 23:00"], tz="UTC"),
    )
    index = pd.DatetimeIndex(
        [first, "2022-01-21 22:00", "2022-01-21 23:00", "2022-01-22 00:00"], tz="UTC"
    )
    prices = pd.Series(1.0, index, name="BG")

    if expected is None:
        with pytest.raises(AssertionError, match="before their first document"):
            currency_per_timestamp(prices, currencies)
    else:
        assert currency_per_timestamp(prices, currencies).tolist() == expected
