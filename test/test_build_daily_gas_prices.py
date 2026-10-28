# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for scripts/build_daily_gas_prices.py."""

import numpy as np
import pandas as pd
import pytest

from scripts.build_daily_gas_prices import (
    gas_days,
    hourly_gas_price,
    read_mibgas_prices,
    read_the_prices,
)

THE_XML = """<?xml version="1.0" encoding="utf-8"?>
<PricesEnergyImbalance xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <PricesEnergyImbalance>
    <Gasday>2023-01-02</Gasday>
    <VTPDailyAveragePrice>70.5</VTPDailyAveragePrice>
    <Unit>EUR/MWh</Unit>
  </PricesEnergyImbalance>
  <PricesEnergyImbalance>
    <Gasday>2023-01-01</Gasday>
    <VTPDailyAveragePrice xsi:nil="true" />
    <Unit>EUR/MWh</Unit>
  </PricesEnergyImbalance>
</PricesEnergyImbalance>
"""


def test_read_the_prices(tmp_path):
    fn = tmp_path / "the.xml"
    fn.write_text(THE_XML)
    prices = read_the_prices(fn)
    assert prices.to_dict() == {pd.Timestamp("2023-01-02"): 70.5}


@pytest.mark.parametrize("layout", ["indices", "indexes"])
def test_read_mibgas_prices(tmp_path, layout):
    days = pd.to_datetime(["2023-01-02", "2023-01-01"])
    if layout == "indices":
        sheet = "Indices"
        df = pd.DataFrame(
            {
                "Delivery day": days.repeat(2),
                "Area": ["ES", "PT", "ES", "PT"],
                "MIBGAS Index\n[EUR/MWh]": [60.0, 61.0, 50.0, 51.0],
                "MIBGAS\nLNG Index\n[EUR/MWh]": 1.0,
            }
        )
    else:
        sheet = "MIBGAS Indexes"
        df = pd.DataFrame(
            {
                "Delivery day": days,
                "MIBGAS PVB Average Price Index Day-Ahead\n[EUR/MWh]": 1.0,
                "MIBGAS-ES Index\n[EUR/MWh]": [60.0, 50.0],
                "MIBGAS-PT Index\n[EUR/MWh]": [61.0, 51.0],
                "MIBGAS\nLNG-ES Index\n[EUR/MWh]": 1.0,
            }
        )
    fn = tmp_path / "mibgas.xlsx"
    with pd.ExcelWriter(fn) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Disclaimer")
        df.to_excel(writer, sheet_name=sheet, index=False)
    prices = read_mibgas_prices(fn)
    np.testing.assert_allclose(prices.loc[["2023-01-01", "2023-01-02"]], [50.0, 60.0])


def test_gas_days_start_at_six_local_time():
    snapshots = pd.to_datetime(
        [
            "2023-01-02 04:00",  # 05:00 CET
            "2023-01-02 05:00",  # 06:00 CET
            "2023-07-02 03:00",  # 05:00 CEST
            "2023-07-02 04:00",  # 06:00 CEST
        ]
    )
    expected = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-07-01", "2023-07-02"])
    assert gas_days(snapshots).equals(pd.DatetimeIndex(expected))


def test_hourly_gas_price_fills_only_the_previous_gas_day():
    snapshots = pd.date_range("2023-01-01", "2023-01-02 23:00", freq="h")
    daily = pd.Series([40.0, 50.0], index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
    price = hourly_gas_price(daily, snapshots)
    np.testing.assert_allclose(price.loc["2023-01-01 00:00":"2023-01-02 04:00"], 40.0)
    np.testing.assert_allclose(price.loc["2023-01-02 05:00":], 50.0)

    with pytest.raises(AssertionError, match="2023-01-02"):
        hourly_gas_price(daily.iloc[:1], snapshots)
