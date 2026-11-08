# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the solar calibration in scripts/lib/backcast_calibration.py."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.lib.backcast_calibration import calibrate_solar, read_solar_unpublished
from scripts.lib.validation.config.backcast import BackcastConfig
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


ES_UNPUBLISHED = pd.Series({"ES": 40e-6})


@pytest.mark.parametrize(
    "observed, exclude, shares, unpublished, expected",
    [
        (ES_SOLAR, [], SHARES, NO_SHARE, {"ES": 0.1, "IE": 0.4}),
        (ES_SOLAR, [], SHARES, ES_UNPUBLISHED, {"ES": 0.15, "IE": 0.4}),
        (ES_SOLAR_JAN, [], NO_SHARE, NO_SHARE, {"ES": 0.05, "IE": 0.5}),
        (ES_SOLAR, ["ES"], SHARES, NO_SHARE, {"ES": 0.35, "IE": 0.4}),
        (ES_SOLAR, ["ES"], NO_SHARE, NO_SHARE, {"ES": 0.5, "IE": 0.5}),
        (ES_ONWIND, [], SHARES, NO_SHARE, {"ES": 0.0, "IE": 0.4}),
    ],
    ids=[
        "calibrated",
        "unpublished",
        "observed-months",
        "excluded-self-consumption",
        "excluded-no-share",
        "published-without-solar",
    ],
)
def test_calibrate_solar(network, observed, exclude, shares, unpublished, expected):
    calibrate_solar(
        network, frame(observed), AREA_TO_BUS, exclude, WIDE, shares, unpublished
    )
    p_max_pu = network.generators_t.p_max_pu
    np.testing.assert_allclose(p_max_pu[ES], expected["ES"])
    np.testing.assert_allclose(p_max_pu[IE], expected["IE"])
    np.testing.assert_allclose(p_max_pu["ES00 onwind"], 0.5)


def test_calibrate_solar_raises_on_partial_shares(network):
    network.buses.loc["ESCE", "country"] = "PT"
    with pytest.raises(
        AssertionError, match=r"missing for some countries of ES: \['PT'\]"
    ):
        calibrate_solar(
            network, frame(ES_ONWIND), AREA_TO_BUS, ["ES"], WIDE, SHARES, NO_SHARE
        )


@pytest.mark.parametrize(
    "exclude, shares, unpublished, match",
    [
        ([], SHARES, NO_SHARE, r"0.20 of ES is outside the bounds \[0.5, 2.0\]"),
        (["XX"], SHARES, NO_SHARE, r"Unknown ENTSO-E areas .* \['XX'\]"),
        ([], pd.Series({"ES": 1.0}), NO_SHARE, r"outside \[0, 1\): \{'ES': 1.0\}"),
        ([], pd.Series({"IE": -0.1}), NO_SHARE, r"outside \[0, 1\): \{'IE': -0.1\}"),
        ([], SHARES, pd.Series({"XX": 1.0}), r"Unknown ENTSO-E areas .* \['XX'\]"),
        ([], SHARES, pd.Series({"IE_SEM": 1.0}), r"uncalibrated areas: \['IE_SEM'\]"),
        (["ES"], SHARES, ES_UNPUBLISHED, r"uncalibrated areas: \['ES'\]"),
    ],
    ids=[
        "out-of-bounds",
        "unknown-excluded-area",
        "share-one",
        "negative-share",
        "unknown-unpublished-area",
        "unpublished-unpublished-area",
        "unpublished-excluded-area",
    ],
)
def test_calibrate_solar_raises(network, exclude, shares, unpublished, match):
    with pytest.raises(AssertionError, match=match):
        calibrate_solar(
            network, frame(ES_SOLAR), AREA_TO_BUS, exclude, BOUNDS, shares, unpublished
        )


def test_calibrate_solar_requires_time_varying_availability(network):
    network.add("Generator", "ES00 solar static", bus="ES00", carrier="solar")
    with pytest.raises(AssertionError, match=r"without time-varying p_max_pu: \['ES00"):
        calibrate_solar(
            network, frame(ES_SOLAR), AREA_TO_BUS, [], WIDE, SHARES, NO_SHARE
        )


@pytest.mark.parametrize(
    "rows, match",
    [
        ("ES,2019,1.0\n", r"No unpublished solar generation for 2020"),
        ("ES,2020,1.0\nES,2020,2.0\n", r"Duplicate areas for 2020 .*\['ES'\]"),
    ],
    ids=["missing-year", "duplicate-area"],
)
def test_read_solar_unpublished_raises(tmp_path, rows, match):
    path = tmp_path / "solar_unpublished.csv"
    path.write_text("area,year,twh,source\n" + rows)
    with pytest.raises(AssertionError, match=match):
        read_solar_unpublished(str(path), 2020)


def test_read_solar_unpublished(tmp_path):
    path = tmp_path / "solar_unpublished.csv"
    path.write_text(
        "area,year,twh,source\nES,2019,1.0,a\nES,2020,2.0,b\nIE_SEM,2020,0.5,c\n"
    )
    unpublished = read_solar_unpublished(str(path), 2020)
    assert unpublished.to_dict() == {"ES": 2.0, "IE_SEM": 0.5}


def test_solar_unpublished_requires_calibration():
    with pytest.raises(ValidationError, match="requires calibrate_solar"):
        BackcastConfig(solar_unpublished=True)
