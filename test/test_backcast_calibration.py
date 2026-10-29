# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for the ENTSO-E actual generation retrieval and build and the calibration in scripts/lib/backcast_calibration.py."""

import datetime

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.build_entsoe_generation import (
    hourly_carrier_generation,
    hourly_generation,
    monthly_generation,
)
from scripts.lib.backcast_calibration import apply_monthly_availability, calibrate_wind
from scripts.lib.validation.config import validate_config
from scripts.lib.validation.config.backcast import BackcastConfig
from scripts.retrieve_actual_generation_entsoe import aggregated_generation

SNAPSHOTS = pd.date_range("2023-01-31 22:00", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"ES00": "ES", "ESCE": "ES", "IE00": "IE_SEM"})
BOUNDS = [0.5, 2.0]
WIDE = [0.0, 100.0]
JAN_END = {"ES": "2023-01-31"}
IE_JAN_END = {"IE_SEM": "2023-01-31"}
IGNORED_PHASE_OUT = r"without calibrated nuclear generators: \['IE_SEM'\]"


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    wind = ["ES00 onwind", "ESCE onwind", "IE00 onwind", "IE00 offwind-ac"]
    n.add(
        "Generator",
        wind,
        bus=["ES00", "ESCE", "IE00", "IE00"],
        carrier=["onwind", "onwind", "onwind", "offwind-ac"],
        p_nom=100.0,
        p_max_pu=pd.DataFrame(0.5, index=SNAPSHOTS, columns=wind),
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


def test_aggregated_generation_drops_consumption_and_converts_to_utc():
    index = pd.date_range("2023-12-31 23:00", periods=3, freq="h", tz="Europe/Berlin")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Solar", "Actual Aggregated"),
            ("Solar", "Actual Consumption"),
            ("Nuclear", "Actual Aggregated"),
        ]
    )
    raw = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=index, columns=columns)
    result = aggregated_generation(raw, pd.Timestamp("2024-01-01", tz="UTC"))
    assert result.columns.tolist() == ["Solar", "Nuclear"]
    assert result.index.equals(index[:2].tz_convert("UTC"))


def test_aggregated_generation_merges_mixed_periods():
    plain = pd.DataFrame(
        {"Nuclear": [1.0, 2.0], "Solar": [3.0, 4.0]},
        index=pd.date_range("2020-09-30 22:00", periods=2, freq="h", tz="UTC"),
    )
    nested = pd.DataFrame(
        [[5.0, 6.0, 0.5]],
        index=pd.date_range("2020-10-01 00:00", periods=1, freq="h", tz="UTC"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("Nuclear", "Actual Aggregated"),
                ("Solar", "Actual Aggregated"),
                ("Solar", "Actual Consumption"),
            ]
        ),
    )
    raw = pd.concat([plain, nested])
    result = aggregated_generation(raw, pd.Timestamp("2021-01-01", tz="UTC"))
    assert result.columns.tolist() == ["Nuclear", "Solar"]
    np.testing.assert_allclose(result.Nuclear, [1, 2, 5])
    np.testing.assert_allclose(result.Solar, [3, 4, 6])


def test_hourly_carrier_generation_maps_production_types():
    index = pd.date_range("2023-01-01", periods=8, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Fossil Brown coal/Lignite": [4.0] * 8,
            "Fossil Oil shale": [np.nan] * 4 + [2.0] * 4,
            "Wind Offshore": [1.0] * 8,
            "Other": [9.0] * 8,
        },
        index=index,
    )
    result = hourly_carrier_generation(raw)
    expected = pd.DataFrame(
        {"lignite": [4.0, 6.0], "offwind": [1.0, 1.0]},
        index=pd.date_range("2023-01-01", periods=2, freq="h"),
    )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_monthly_generation_sums_within_snapshots():
    hourly = pd.DataFrame(
        {
            "nuclear": [1.0, 2.0, np.nan, 4.0, 8.0],
            "onwind": [5.0, np.nan, np.nan, 3.0, np.nan],
        },
        index=pd.date_range("2023-01-31 21:00", periods=5, freq="h"),
    )
    result = monthly_generation(hourly_generation({"ES": hourly}, SNAPSHOTS))
    expected = frame(
        {
            ("ES", "nuclear", "2023-01-01"): (2.0, 1),
            ("ES", "nuclear", "2023-02-01"): (12.0, 2),
            ("ES", "onwind", "2023-02-01"): (3.0, 1),
        }
    )
    pd.testing.assert_frame_equal(result, expected, check_index_type=False)


ES_FULL = {
    ("ES", "onwind", "2023-01-01"): (40.0, 2),
    ("ES", "onwind", "2023-02-01"): (40.0, 2),
}
ES_JAN = {("ES", "onwind", "2023-01-01"): (40.0, 2)}
ES_JAN_PARTIAL = {("ES", "onwind", "2023-01-01"): (20.0, 1)}
IE_JAN = {("IE_SEM", "onwind", "2023-01-01"): (100.0, 2)}
IE_OFF = {"IE00 offwind-ac": 0.0}
IE_KEEP = {"IE00 offwind-ac": 0.5}
ES_ON = {"ES": {"onwind": 1.0}}


@pytest.mark.parametrize(
    "observed, curtailment, exclude, expected",
    [
        (ES_FULL | IE_JAN, {}, {}, {"ES00 onwind": 0.1, "IE00 onwind": 0.5} | IE_OFF),
        (ES_JAN | IE_JAN, {}, {}, {"ES00 onwind": 0.1} | IE_OFF),
        (ES_JAN_PARTIAL, {}, {}, {"ES00 onwind": 0.1, "IE00 onwind": 0.5} | IE_KEEP),
        (ES_FULL, {"ES": 0.00008}, {}, {"ES00 onwind": 0.2} | IE_KEEP),
        (ES_FULL | IE_JAN, {"IE_SEM": 0.001}, {}, {"IE00 onwind": 1.0} | IE_OFF),
        (
            ES_FULL | IE_JAN,
            {"IE_SEM": {"onwind": 0.00005}},
            {},
            {"IE00 onwind": 0.75} | IE_OFF,
        ),
        (
            ES_FULL | IE_JAN,
            {},
            {"ES": ["onwind"]},
            {"ES00 onwind": 0.5, "IE00 onwind": 0.5} | IE_OFF,
        ),
        (ES_FULL, {}, {"ES": ["offwind"]}, {"ES00 onwind": 0.1} | IE_KEEP),
        (ES_FULL | IE_JAN, {}, {"IE_SEM": ["offwind"]}, IE_KEEP),
    ],
    ids=[
        "all",
        "observed-months",
        "partial-month",
        "curtailment",
        "clip",
        "per-carrier",
        "excluded",
        "excluded-other-carrier",
        "excluded-unpublished-carrier",
    ],
)
def test_calibrate_wind(network, observed, curtailment, exclude, expected):
    calibrate_wind(
        network, frame(observed), AREA_TO_BUS, curtailment, exclude, WIDE, {}
    )
    p_max_pu = network.generators_t.p_max_pu
    for gen, value in expected.items():
        assert np.allclose(p_max_pu[gen], value)
    assert np.allclose(p_max_pu["ESCE onwind"], p_max_pu["ES00 onwind"])


@pytest.mark.parametrize(
    "target, exclude, expected",
    [
        ({"onwind": 0.0001}, {}, {"IE00 onwind": 0.25, "IE00 offwind-ac": 0.5}),
        (
            {"onwind": 0.0001, "offwind": 0.0004},
            {},
            {"IE00 onwind": 0.25, "IE00 offwind-ac": 1.0},
        ),
        (
            {"onwind": 0.0001},
            {"IE_SEM": ["offwind"]},
            {"IE00 onwind": 0.25, "IE00 offwind-ac": 0.5},
        ),
    ],
    ids=["onwind", "both-clipped", "other-carrier-excluded"],
)
def test_calibrate_wind_target(network, target, exclude, expected):
    calibrate_wind(
        network, frame(ES_FULL), AREA_TO_BUS, {}, exclude, WIDE, {"IE_SEM": target}
    )
    p_max_pu = network.generators_t.p_max_pu
    for gen, value in expected.items():
        np.testing.assert_allclose(p_max_pu[gen], value)


def test_calibrate_wind_pools_offshore_carriers(network):
    offwind = ["IE00 offwind-dc", "IE00 offwind-float"]
    network.add(
        "Generator",
        offwind,
        bus="IE00",
        carrier=["offwind-dc", "offwind-float"],
        p_nom=[100.0, 200.0],
        p_max_pu=pd.DataFrame([[0.5, 0.25]] * 4, index=SNAPSHOTS, columns=offwind),
    )
    observed = frame({("IE_SEM", "offwind", "2023-01-01"): (240.0, 2)})
    calibrate_wind(network, observed, AREA_TO_BUS, {}, {}, BOUNDS, {})
    p_max_pu = network.generators_t.p_max_pu
    for gen, value in {
        "IE00 offwind-ac": 0.4,
        **dict(zip(offwind, [0.4, 0.2])),
    }.items():
        np.testing.assert_allclose(p_max_pu[gen], value)


@pytest.mark.parametrize(
    "observed, curtailment, exclude, target, match",
    [
        (
            {("ES", "nuclear", "2023-01-01"): (1.0, 2)},
            {"ES": 1.0},
            {},
            {},
            "without actual",
        ),
        (ES_FULL, {"ES": {"offwind": 1.0}}, {}, {}, "without actual"),
        (
            {("ES", "offwind", "2023-01-01"): (1.0, 2)},
            {},
            {},
            {},
            "offwind generation in ES",
        ),
        (
            ES_FULL,
            {},
            {},
            {},
            r"0.20 of onwind in ES is outside the bounds \[0.5, 2.0\]",
        ),
        (ES_FULL, {"XX": 1.0}, {}, {}, r"Unknown ENTSO-E areas .* \['XX'\]"),
        (ES_FULL, {}, {"XX": ["onwind"]}, {}, r"Unknown ENTSO-E areas .* \['XX'\]"),
        (ES_FULL, {"ES": 1.0}, {"ES": ["offwind"]}, {}, "areas with excluded carriers"),
        (ES_FULL, {}, {}, ES_ON, r"published generation: \['ES'\]"),
        (
            IE_JAN,
            {},
            {"ES": ["onwind"]},
            ES_ON,
            r"excluded carriers: \[\('ES', 'onwind'\)\]",
        ),
        (
            ES_FULL,
            {},
            {},
            {"XX": {"onwind": 1.0}},
            r"Unknown ENTSO-E areas .* \['XX'\]",
        ),
    ],
    ids=[
        "curtailment-no-wind",
        "curtailment-no-carrier",
        "no-generators",
        "out-of-bounds",
        "unknown-curtailment-area",
        "unknown-excluded-area",
        "curtailment-excluded",
        "target-published",
        "target-excluded",
        "unknown-target-area",
    ],
)
def test_calibrate_wind_raises(network, observed, curtailment, exclude, target, match):
    with pytest.raises(AssertionError, match=match):
        calibrate_wind(
            network, frame(observed), AREA_TO_BUS, curtailment, exclude, BOUNDS, target
        )


ES_NUCLEAR_JAN = {("ES", "nuclear", "2023-01-01"): (400.0, 2)}


@pytest.mark.parametrize(
    "observed, phase_out, expected",
    [
        (
            ES_NUCLEAR_JAN | {("ES", "nuclear", "2023-02-01"): (2000.0, 2)},
            {},
            [0.5, 0.5, 1.0, 1.0],
        ),
        (
            {
                ("ES", "nuclear", "2023-01-01"): (200.0, 1),
                ("ES", "nuclear", "2023-02-01"): (400.0, 2),
            },
            {},
            [0.5] * 4,
        ),
        (ES_NUCLEAR_JAN, JAN_END, [0.5, 0.5, 0.0, 0.0]),
        ({("IE_SEM", "onwind", "2023-01-01"): (1.0, 2)}, {}, [0.9] * 4),
    ],
    ids=["clipped", "partial-hours", "phase-out", "unpublished-area"],
)
def test_apply_monthly_availability(network, observed, phase_out, expected):
    apply_monthly_availability(
        network, frame(observed), AREA_TO_BUS, "nuclear", {}, phase_out
    )
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in ["ES00 nuclear", "ESCE nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], expected)


def test_apply_monthly_availability_keep_unpublished(network, caplog):
    caplog.set_level("INFO")
    apply_monthly_availability(
        network,
        frame({("ES", "onwind", "2023-01-01"): (1.0, 2)}),
        AREA_TO_BUS,
        "nuclear",
        {},
        {},
        keep_unpublished=True,
    )
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in ["ES00 nuclear", "ESCE nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], 0.9)
    assert "No actual nuclear generation published for ['ES']" in caplog.text


def test_apply_monthly_availability_exclude(network, caplog):
    caplog.set_level("INFO")
    apply_monthly_availability(
        network,
        frame(ES_NUCLEAR_JAN | {("ES", "nuclear", "2023-02-01"): (2000.0, 2)}),
        AREA_TO_BUS,
        "nuclear",
        {},
        {},
        exclude=["ES"],
    )
    p_max_pu = network.get_switchable_as_dense("Generator", "p_max_pu")
    for gen in ["ES00 nuclear", "ESCE nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], 0.9)
    assert "Excluded ['ES'] from the monthly nuclear availability" in caplog.text


def test_apply_monthly_availability_exclude_raises(network):
    with pytest.raises(AssertionError, match="Unknown ENTSO-E"):
        apply_monthly_availability(
            network, frame(ES_NUCLEAR_JAN), AREA_TO_BUS, "nuclear", {}, {}, exclude=["XX"]
        )


@pytest.mark.parametrize(
    "observed, area_map, phase_out, match",
    [
        (
            {("ES", "onwind", "2023-01-01"): (1.0, 2)},
            {},
            {},
            "No actual nuclear generation for ES",
        ),
        (ES_NUCLEAR_JAN, {}, {}, r"for ES in \['2023-02'\] before its phase-out"),
        (ES_NUCLEAR_JAN, {}, {"XX": "2023-01-31"}, "Unknown ENTSO-E"),
        (ES_NUCLEAR_JAN, {"IE_SEM": "XX"}, JAN_END, "Unknown ENTSO-E"),
        (ES_NUCLEAR_JAN, {}, JAN_END | IE_JAN_END, IGNORED_PHASE_OUT),
        (ES_NUCLEAR_JAN, {"IE_SEM": "ES"}, JAN_END | IE_JAN_END, IGNORED_PHASE_OUT),
    ],
    ids=[
        "no-data",
        "missing-month",
        "unknown-phase-out",
        "unknown-map",
        "phase-out-without-generators",
        "phase-out-mapped-area",
    ],
)
def test_apply_monthly_availability_raises(
    network, observed, area_map, phase_out, match
):
    with pytest.raises(AssertionError, match=match):
        apply_monthly_availability(
            network, frame(observed), AREA_TO_BUS, "nuclear", area_map, phase_out
        )


@pytest.mark.parametrize(
    "carrier, calibrate",
    [
        ("onwind", lambda n, g: calibrate_wind(n, g, AREA_TO_BUS, {}, {}, WIDE, {})),
        (
            "nuclear",
            lambda n, g: apply_monthly_availability(
                n, g, AREA_TO_BUS, "nuclear", {}, JAN_END
            ),
        ),
    ],
)
def test_calibration_fails_on_unmapped_generators(network, carrier, calibrate):
    network.add("Bus", "XX00", carrier="AC")
    network.add("Generator", f"XX00 {carrier}", bus="XX00", carrier=carrier)
    with pytest.raises(AssertionError, match=f"without ENTSO-E area: .*XX00 {carrier}"):
        calibrate(network, frame(ES_FULL | ES_NUCLEAR_JAN))


@pytest.mark.parametrize(
    "calibrate",
    [
        lambda n: calibrate_wind(
            n, frame(ES_FULL | IE_JAN), AREA_TO_BUS, {"IE_SEM": 0.001}, {}, WIDE, {}
        ),
        lambda n: apply_monthly_availability(
            n,
            frame(ES_NUCLEAR_JAN | {("ES", "nuclear", "2023-02-01"): (2000.0, 2)}),
            AREA_TO_BUS,
            "nuclear",
            {},
            {},
        ),
    ],
    ids=["wind", "nuclear"],
)
def test_calibration_warns_on_clipping(network, caplog, calibrate):
    calibrate(network)
    assert "Clipping loses more than 1%" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        {"wind_curtailment_twh": {"DE_LU": -1.0}},
        {"wind_curtailment_twh": {"DE_LU": {"solar": 1.0}}},
        {"hybrid_capacities": {"sources": ["offered_capacity"]}},
        {"hybrid_capacities": {"physical_factor": 0.0}},
        {"hybrid_capacities": {"sources": []}},
        {"wind_calibration_factor_bounds": [2.0, 0.5]},
        {"wind_calibration_factor_bounds": [0.0, 2.0]},
        {"wind_calibration_factor_bounds": [0.5]},
        {"nuclear_phase_out": {"DE_LU": "April"}},
        {"nuclear_phase_out": {"DE_LU": datetime.date(2023, 4, 15)}},
        {"wind_target_twh": {"GB": {"onwind": 1.0}}},
        {"wind_target_twh": {"GB": {"solar": 1.0}}, "exclude_generation_areas": ["GB"]},
        {
            "wind_target_twh": {"GB": {"onwind": -1.0}},
            "exclude_generation_areas": ["GB"],
        },
    ],
)
def test_backcast_config_rejects_invalid_calibration(config):
    with pytest.raises(ValidationError):
        BackcastConfig(**config)


def test_backcast_config_accepts_per_carrier_curtailment():
    curtailment = {"DE_LU": {"onwind": 6.0, "offwind": 4.0}, "DK_1": 1.0}
    config = BackcastConfig(wind_curtailment_twh=curtailment)
    assert config.wind_curtailment_twh == curtailment


@pytest.mark.parametrize("foresight", ["myopic", "perfect"])
def test_backcast_calibration_requires_overnight_foresight(foresight):
    backcast = {"calibrate_wind": True}
    validate_config({"backcast": backcast})
    with pytest.raises(ValueError, match="require foresight: overnight"):
        validate_config({"foresight": foresight, "backcast": backcast})


def test_hybrid_capacities_require_ntc():
    with pytest.raises(ValueError, match="requires electricity: cross_border"):
        validate_config({"backcast": {"hybrid_capacities": {"enable": True}}})


def test_apply_monthly_availability_pools_mapped_areas(network):
    network.add("Generator", "IE00 nuclear", bus="IE00", carrier="nuclear", p_nom=400.0)
    observed = frame({("ES", "nuclear", "2023-01-01"): (800.0, 2)})
    apply_monthly_availability(
        network, observed, AREA_TO_BUS, "nuclear", {"IE_SEM": "ES"}, JAN_END
    )
    p_max_pu = network.generators_t.p_max_pu
    for gen in ["ES00 nuclear", "ESCE nuclear", "IE00 nuclear"]:
        np.testing.assert_allclose(p_max_pu[gen], [0.5, 0.5, 0.0, 0.0])
