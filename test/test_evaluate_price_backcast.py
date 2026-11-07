# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for evaluate_price_backcast.py."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_pdf import PdfPages

from scripts.evaluate_price_backcast import PLOTS, plot_page
from scripts.lib.backcast_data import (
    align_prices,
    area_metrics,
    duration_curve,
    hourly_area_bias,
    monthly_bias,
    price_metrics,
    summary_metrics,
    zone_group_metrics,
)


@pytest.fixture
def index():
    return pd.date_range("2023-01-31 22:00", periods=4, freq="h")


@pytest.fixture
def observed(index):
    return pd.DataFrame(
        {"DE00": [10.0, 20, 0, -10], "FR00": [1.0, np.nan, 3, 4]}, index
    )


@pytest.fixture
def model(index):
    return pd.DataFrame(
        {"DE00": [12.0, 18, 0, 0], "FR00": [2.0, 50, 6, 8], "XX00": 0.0}, index
    )


def test_align_prices_masks_missing_observed_hours(model, observed):
    aligned_model, aligned_observed = align_prices(model.iloc[1:], observed)

    assert aligned_model.columns.equals(observed.columns)
    assert aligned_model.index.equals(observed.index[1:])
    assert aligned_model.notna().equals(aligned_observed.notna())


@pytest.mark.parametrize(
    "tz_model, tz_observed",
    [("UTC", None), (None, "UTC")],
)
def test_align_prices_rejects_timezone_aware(model, observed, tz_model, tz_observed):
    with pytest.raises(AssertionError, match="naive UTC"):
        align_prices(model.tz_localize(tz_model), observed.tz_localize(tz_observed))


@pytest.mark.parametrize(
    "transform, match",
    [
        (lambda m: m.drop(columns="FR00"), "missing in model"),
        (lambda m: m.shift(10, freq="h"), "no hours"),
    ],
    ids=["bus", "hours"],
)
def test_align_prices_fails(model, observed, transform, match):
    with pytest.raises(AssertionError, match=match):
        align_prices(transform(model), observed)


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("mean_bias", {"DE00": 2.5, "FR00": 8 / 3}),
        ("mae", {"DE00": 3.5, "FR00": 8 / 3}),
        ("rmse", {"DE00": np.sqrt(27.0), "FR00": np.sqrt(26 / 3)}),
        (
            "std_ratio",
            {"DE00": np.std([12, 18, 0, 0]) / np.std([10, 20, 0, -10]), "FR00": 2.0},
        ),
        ("mean_observed", {"DE00": 5.0, "FR00": 8 / 3}),
        ("mean_model", {"DE00": 7.5, "FR00": 16 / 3}),
        (
            "std_observed",
            {
                "DE00": np.std([10, 20, 0, -10], ddof=1),
                "FR00": np.std([1, 3, 4], ddof=1),
            },
        ),
        ("zero_share_model", {"DE00": 0.5, "FR00": 0.0}),
        ("zero_share_observed", {"DE00": 0.25, "FR00": 0.0}),
        ("negative_share_model", {"DE00": 0.0, "FR00": 0.0}),
        ("negative_share_observed", {"DE00": 0.25, "FR00": 0.0}),
    ],
)
def test_price_metrics(model, observed, metric, expected):
    metrics = price_metrics(*align_prices(model, observed))

    pd.testing.assert_series_equal(
        metrics[metric], pd.Series(expected, name=metric).rename_axis("bus")
    )


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_price_metrics_pearson_r_invariant_to_offset(observed, offset):
    metrics = price_metrics(*align_prices(observed + offset, observed))

    np.testing.assert_allclose(metrics["pearson_r"], 1.0)


@pytest.mark.parametrize(
    "model_price, expected",
    [(600.0, 0.0), (-200.0, 600 / np.sqrt(2)), (400.0, 100 / np.sqrt(2))],
)
def test_price_metrics_rmse_clipped(model_price, expected):
    index = pd.date_range("2023-01-01", periods=2, freq="h")
    observed = pd.DataFrame({"DE00": [500.0, -100]}, index)
    model = pd.DataFrame({"DE00": [model_price, -100]}, index)

    metrics = price_metrics(model, observed)

    assert metrics.loc["DE00", "rmse_clipped"] == pytest.approx(expected)


def test_monthly_bias_separates_years(model, observed):
    index = pd.date_range("2022-12-31 22:00", periods=4, freq="h")
    bias = monthly_bias(*align_prices(model.set_axis(index), observed.set_axis(index)))

    expected = pd.DataFrame(
        {"DE00": [0.0, 5.0], "FR00": [1.0, 3.5]},
        index=pd.PeriodIndex(["2022-12", "2023-01"], freq="M", name="month"),
    )
    pd.testing.assert_frame_equal(bias, expected)


def test_duration_curve_sorts_descending_and_drops_missing():
    curve = duration_curve(pd.Series([1.0, np.nan, 3, 2]))

    pd.testing.assert_series_equal(
        curve, pd.Series([3.0, 2, 1], index=[1 / 3, 2 / 3, 1])
    )


@pytest.mark.parametrize("title", PLOTS)
def test_plot_page(model, observed, title, tmp_path):
    path = tmp_path / "plots.pdf"
    with PdfPages(path) as pdf:
        plot_page(pdf, title, PLOTS[title], *align_prices(model, observed))

    assert path.stat().st_size > 0


@pytest.fixture
def area_to_bus():
    return pd.Series({"DE00": "DE_LU", "LUG1": "DE_LU", "FR00": "FR", "ES00": "ES"})


@pytest.fixture
def bus_metrics():
    return pd.DataFrame(
        {
            "mae": [2.0, 4, 1, 10],
            "mean_bias": [-2.0, 4, 1, -6],
            "pearson_r": [0.8, 0.6, 0.9, 0.5],
            "std_ratio": [0.5, 0.7, 1.0, 0.9],
            "zero_share_model": [0.0, 0, 0.1, 0],
            "negative_share_model": [0.1, 0.1, 0, 0],
            "zero_share_observed": [0.0, 0, 0, 0.2],
            "negative_share_observed": [0.2, 0.2, 0, 0],
        },
        index=pd.Index(["DE00", "LUG1", "FR00", "ES00"], name="bus"),
    )


def test_area_metrics_counts_each_area_once(bus_metrics, area_to_bus):
    metrics = area_metrics(bus_metrics, area_to_bus)

    pd.testing.assert_series_equal(
        metrics.mae,
        pd.Series({"DE_LU": 3.0, "ES": 10, "FR": 1}, name="mae").rename_axis("area"),
    )


def test_area_metrics_fails_on_unmapped_bus(bus_metrics, area_to_bus):
    with pytest.raises(KeyError):
        area_metrics(bus_metrics, area_to_bus.drop("LUG1"))


@pytest.mark.parametrize(
    "key, expected",
    [
        ("median_mae", 3.0),
        ("mean_mae", 14 / 3),
        ("pearson_r", 0.7),
        ("abs_bias", 1.0),
        ("std_ratio", 0.9),
        ("share_le0_model", 0.2 / 3),
        ("share_le0_observed", 0.4 / 3),
        ("count", 3),
    ],
)
def test_summary_metrics(bus_metrics, area_to_bus, key, expected):
    summary = summary_metrics(area_metrics(bus_metrics, area_to_bus))

    assert summary[key] == pytest.approx(expected)


def test_zone_group_metrics(bus_metrics, area_to_bus):
    groups = zone_group_metrics(area_metrics(bus_metrics, area_to_bus))

    expected = pd.DataFrame(
        {"mae": [2.0, 10], "mean_bias": [1.0, -6], "pearson_r": [0.8, 0.5]},
        index=pd.Index(["Central West", "Iberia"], name="group"),
    ).assign(count=[2, 1])
    pd.testing.assert_frame_equal(groups, expected)


def test_zone_group_metrics_rejects_unknown_area(bus_metrics, area_to_bus):
    metrics = area_metrics(bus_metrics, area_to_bus).rename(index={"ES": "XX"})

    with pytest.raises(AssertionError, match=r"\['XX'\]"):
        zone_group_metrics(metrics)


def test_hourly_area_bias(area_to_bus):
    index = pd.date_range("2023-01-01", periods=48, freq="h")
    observed = pd.DataFrame(50.0, index, columns=area_to_bus.index)
    model = observed + pd.Series({"DE00": 2.0, "LUG1": 4, "FR00": -1, "ES00": 5})

    bias = hourly_area_bias(model, observed, area_to_bus)

    pd.testing.assert_series_equal(
        bias,
        pd.Series(3.0, pd.RangeIndex(24, name="hour"), name="bias"),
        check_index_type=False,
    )
