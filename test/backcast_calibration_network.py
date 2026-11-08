# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Shared network and generation helpers for the backcast calibration tests."""

import pandas as pd
import pypsa

SNAPSHOTS = pd.date_range("2023-01-31 22:00", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"ES00": "ES", "ESCE": "ES", "IE00": "IE_SEM"})
BOUNDS = [0.5, 2.0]
WIDE = [0.0, 100.0]
NO_SHARE = pd.Series(dtype=float)


def build_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC", country=["ES", "ES", "IE"])
    renewable = ["ES00 onwind", "ESCE onwind", "IE00 onwind", "IE00 offwind-ac"]
    renewable += ["ES00 solar", "ESCE solar rooftop", "IE00 solar-hsat"]
    n.add(
        "Generator",
        renewable,
        bus=["ES00", "ESCE", "IE00", "IE00", "ES00", "ESCE", "IE00"],
        carrier=["onwind", "onwind", "onwind", "offwind-ac"]
        + ["solar", "solar rooftop", "solar-hsat"],
        p_nom=100.0,
        p_max_pu=pd.DataFrame(0.5, index=SNAPSHOTS, columns=renewable),
    )
    n.add(
        "Generator",
        ["ES00 nuclear", "ESCE nuclear"],
        bus=["ES00", "ESCE"],
        carrier="nuclear",
        p_nom=[100.0, 300.0],
        p_max_pu=0.9,
    )
    return n


def frame(entries: dict[tuple[str, str, str], tuple[float, int]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(a, c, pd.Timestamp(m)) for (a, c, m) in entries],
        names=["area", "carrier", "month"],
    )
    return pd.DataFrame(list(entries.values()), index, ["generation", "hours"])
