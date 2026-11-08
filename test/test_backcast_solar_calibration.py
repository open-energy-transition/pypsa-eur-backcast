# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the solar calibration in scripts/lib/backcast_calibration.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.lib.backcast_calibration import calibrate_solar
from test.backcast_calibration_network import (
    AREA_TO_BUS,
    BOUNDS,
    NO_SHARE,
    WIDE,
    build_network,
    frame,
)

ES_SOLAR = {
    ("ES", "solar", "2023-01-01"): (40.0, 2),
    ("ES", "solar", "2023-02-01"): (40.0, 2),
}
ES_SOLAR_JAN = {("ES", "solar", "2023-01-01"): (20.0, 2)}
ES_ONWIND = {("ES", "onwind", "2023-01-01"): (40.0, 2)}
SHARES = pd.Series({"ES": 0.3, "IE": 0.2})
ES = ["ES00 solar", "ESCE solar rooftop"]
IE = ["IE00 solar-hsat"]


@pytest.fixture
def network() -> pypsa.Network:
    return build_network()


@pytest.mark.parametrize(
    "observed, exclude, shares, expected",
    [
        (ES_SOLAR, [], SHARES, {"ES": 0.1, "IE": 0.4}),
        (ES_SOLAR_JAN, [], NO_SHARE, {"ES": 0.05, "IE": 0.5}),
        (ES_SOLAR, ["ES"], SHARES, {"ES": 0.35, "IE": 0.4}),
        (ES_SOLAR, ["ES"], NO_SHARE, {"ES": 0.5, "IE": 0.5}),
        (ES_ONWIND, [], SHARES, {"ES": 0.0, "IE": 0.4}),
    ],
    ids=[
        "calibrated",
        "observed-months",
        "excluded-self-consumption",
        "excluded-no-share",
        "published-without-solar",
    ],
)
def test_calibrate_solar(network, observed, exclude, shares, expected):
    calibrate_solar(network, frame(observed), AREA_TO_BUS, exclude, WIDE, shares)
    p_max_pu = network.generators_t.p_max_pu
    np.testing.assert_allclose(p_max_pu[ES], expected["ES"])
    np.testing.assert_allclose(p_max_pu[IE], expected["IE"])
    np.testing.assert_allclose(p_max_pu["ES00 onwind"], 0.5)


def test_calibrate_solar_raises_on_partial_shares(network):
    network.buses.loc["ESCE", "country"] = "PT"
    with pytest.raises(
        AssertionError, match=r"missing for some countries of ES: \['PT'\]"
    ):
        calibrate_solar(network, frame(ES_ONWIND), AREA_TO_BUS, ["ES"], WIDE, SHARES)


@pytest.mark.parametrize(
    "exclude, shares, match",
    [
        ([], SHARES, r"0.20 of ES is outside the bounds \[0.5, 2.0\]"),
        (["XX"], SHARES, r"Unknown ENTSO-E areas .* \['XX'\]"),
        ([], pd.Series({"ES": 1.0}), r"outside \[0, 1\): \{'ES': 1.0\}"),
        ([], pd.Series({"IE": -0.1}), r"outside \[0, 1\): \{'IE': -0.1\}"),
    ],
    ids=["out-of-bounds", "unknown-excluded-area", "share-one", "negative-share"],
)
def test_calibrate_solar_raises(network, exclude, shares, match):
    with pytest.raises(AssertionError, match=match):
        calibrate_solar(network, frame(ES_SOLAR), AREA_TO_BUS, exclude, BOUNDS, shares)


def test_calibrate_solar_requires_time_varying_availability(network):
    network.add("Generator", "ES00 solar static", bus="ES00", carrier="solar")
    with pytest.raises(AssertionError, match=r"without time-varying p_max_pu: \['ES00"):
        calibrate_solar(network, frame(ES_SOLAR), AREA_TO_BUS, [], WIDE, SHARES)
