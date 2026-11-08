# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for build_cross_border_capacities.py and the cross-border link replacement in scripts/lib/cross_border.py."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.build_cross_border_capacities import (
    area_capacities,
    border_shares,
    build_capacities,
    scheduled_exchanges,
)
from scripts.compose_network import apply_external_flows
from scripts.lib.cross_border import (
    add_external_flows,
    apply_cross_border_capacities,
    external_net_imports,
    inter_zone_branches,
    raise_to_physical,
)

SNAPSHOTS = pd.date_range("2023-01-31 22:00", periods=4, freq="h")
AREA_TO_BUS = pd.Series({"DE00": "DE_LU", "LUG1": "DE_LU", "FR00": "FR"})
AREAS = ["DE_LU", "FR"]


@pytest.fixture
def network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(SNAPSHOTS)
    n.add("Bus", AREA_TO_BUS.index, carrier="AC")
    n.add(
        "Line",
        ["DE-FR", "LU-FR", "LU-FR-planned"],
        bus0=["FR00", "LUG1", "LUG1"],
        bus1=["DE00", "FR00", "FR00"],
        s_nom=[3000.0, 1000.0, 0.0],
        x=0.1,
        length=[400.0, 200.0, 100.0],
    )
    n.add(
        "Link",
        "copper_DE00_LUG1_0",
        bus0="DE00",
        bus1="LUG1",
        carrier="copper",
        p_nom=np.inf,
        p_min_pu=-1,
        underwater_fraction=0.0,
    )
    return n


def retrieved(area_from: str, area_to: str, **columns: list[float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[area_from], [area_to], SNAPSHOTS],
        names=["area_from", "area_to", "utc_timestamp"],
    )
    data = {c: columns.get(c, [np.nan] * 4) for c in ["offered_capacity", "ntc"]}
    data["scheduled_exchange"] = columns.get("scheduled_exchange", [0.0] * 4)
    return pd.DataFrame(data, index=index)


def test_border_shares_orient_and_split(network):
    pairs = border_shares(inter_zone_branches(network, AREA_TO_BUS)).set_index(
        ["bus0", "bus1"]
    )

    assert pairs.index.tolist() == [("DE00", "FR00"), ("FR00", "LUG1")]
    assert pairs[["area0", "area1"]].values.tolist() == [
        ["DE_LU", "FR"],
        ["FR", "DE_LU"],
    ]
    np.testing.assert_allclose(pairs.share, [0.75, 0.25])


@pytest.mark.parametrize(
    "columns, reverse, proxy, capacity, source",
    [
        (
            dict(ntc=[10, 20, 30, 40], offered_capacity=[90, 90, 90, 90]),
            [0, 0, 0, 0],
            [],
            [10, 20, 30, 40],
            "ntc",
        ),
        (
            dict(ntc=[10, np.nan, np.nan, 40], offered_capacity=[90, 90, 90, 90]),
            [0, 0, 0, 0],
            [],
            [10, 20, 30, 40],
            "ntc",
        ),
        (
            dict(offered_capacity=[np.nan, 9, 9, np.nan]),
            [0, 0, 0, 0],
            [],
            [9, 9, 9, 9],
            "offered_capacity",
        ),
        (dict(), [0, 0, 0, 0], [], [6, 6, 7, 7], "scheduled_exchange"),
        (dict(), [0, 8, 0, 0], [], [8, 8, 7, 7], "scheduled_exchange"),
        (
            dict(ntc=[90, 90, 90, 90]),
            [0, 0, 0, 0],
            ["ES-FR"],
            [6, 6, 7, 7],
            "scheduled_exchange",
        ),
        (dict(ntc=[1, 1, 1, 1]), [0, 0, 0, 0], [], [5, 6, 7, 2], "ntc"),
    ],
    ids=[
        "ntc-first",
        "interpolated",
        "edge-gaps",
        "monthly-max-exchange",
        "monthly-max-exchange-both-directions",
        "exchange-proxy-border",
        "raised-to-exchange",
    ],
)
def test_area_capacities_source_priority(columns, reverse, proxy, capacity, source):
    data = pd.concat(
        [
            retrieved("FR", "ES", scheduled_exchange=[5, 6, 7, 2], **columns),
            retrieved("ES", "FR", scheduled_exchange=reverse),
        ]
    )

    capacities, info = area_capacities(
        data, [("FR", "ES")], SNAPSHOTS, max_gap=2, exchange_proxy_borders=proxy
    )

    np.testing.assert_allclose(capacities["FR->ES"], capacity)
    assert info.loc["FR->ES", "source"] == source


def test_area_capacities_exchange_proxy_fallback():
    data = pd.concat(
        [
            retrieved(
                "FR",
                "ES",
                offered_capacity=[9, 9, np.nan, np.nan],
                scheduled_exchange=[5, 6, 7, 2],
            ),
            retrieved("ES", "FR"),
        ]
    )

    with pytest.raises(AssertionError, match="FR->ES"):
        area_capacities(
            data, [("FR", "ES")], SNAPSHOTS, max_gap=1, exchange_proxy_borders=[]
        )
    capacities, info = area_capacities(
        data,
        [("FR", "ES")],
        SNAPSHOTS,
        max_gap=1,
        exchange_proxy_borders=[],
        exchange_proxy_fallback=True,
    )

    np.testing.assert_allclose(capacities["FR->ES"], [9, 9, 7, 7])
    assert info.loc["FR->ES", "source"] == "offered_capacity+scheduled_exchange"
    assert info.loc["ES->FR", "source"] == "scheduled_exchange"


def test_area_capacities_exchange_proxy_by_direction():
    data = pd.concat(
        [
            retrieved("NO_1", "NO_3", scheduled_exchange=[3, 0, 0, 1]),
            retrieved("NO_3", "NO_1", scheduled_exchange=[0, 2000, 2900, 0]),
        ]
    )

    both, _ = area_capacities(
        data, [("NO_1", "NO_3")], SNAPSHOTS, max_gap=1, exchange_proxy_borders=[]
    )
    by_direction, info = area_capacities(
        data,
        [("NO_1", "NO_3")],
        SNAPSHOTS,
        max_gap=1,
        exchange_proxy_borders=[],
        exchange_proxy_by_direction_areas=["NO_1", "NO_3"],
    )

    np.testing.assert_allclose(both["NO_1->NO_3"], [2000, 2000, 2900, 2900])
    np.testing.assert_allclose(by_direction["NO_1->NO_3"], [3, 3, 1, 1])
    np.testing.assert_allclose(by_direction["NO_3->NO_1"], [2000, 2000, 2900, 2900])
    assert info.loc["NO_1->NO_3", "source"] == "scheduled_exchange"


def test_area_capacities_resample_quarter_hours():
    index = pd.MultiIndex.from_product(
        [["FR"], ["ES"], pd.date_range(SNAPSHOTS[0], periods=16, freq="15min")],
        names=["area_from", "area_to", "utc_timestamp"],
    )
    data = pd.concat(
        [
            pd.DataFrame(
                {"offered_capacity": np.repeat([1.0, 2, 3, 4], 4), "ntc": np.nan},
                index=index,
            ).assign(scheduled_exchange=0.0),
            retrieved("ES", "FR"),
        ]
    )

    capacities, info = area_capacities(
        data, [("ES", "FR")], SNAPSHOTS, max_gap=0, exchange_proxy_borders=[]
    )

    np.testing.assert_allclose(capacities["FR->ES"], [1, 2, 3, 4])
    assert info.loc["FR->ES", "source"] == "offered_capacity"


def test_build_capacities_maps_directions_to_buses(network):
    pairs = border_shares(inter_zone_branches(network, AREA_TO_BUS))
    data = pd.concat(
        [
            retrieved("DE_LU", "FR", offered_capacity=[400, 400, 400, 400]),
            retrieved("FR", "DE_LU", ntc=[800, np.nan, 800, 800]),
        ]
    )

    capacities, sources = build_capacities(
        data, pairs, SNAPSHOTS, {}, max_gap=1, exchange_proxy_borders=[]
    )

    expected = {
        "DE00->FR00": 300,
        "FR00->DE00": 600,
        "FR00->LUG1": 200,
        "LUG1->FR00": 100,
    }
    assert sorted(capacities.columns) == sorted(expected)
    for column, value in expected.items():
        np.testing.assert_allclose(capacities[column], value)
    assert sources.loc["LUG1->FR00", "area_from"] == "DE_LU"
    assert sources.loc["FR00->LUG1", "source"] == "ntc"
    assert sources.loc["FR00->LUG1", "interpolated_hours"] == 1


@pytest.mark.parametrize(
    "data",
    [
        retrieved("DE_LU", "FR", offered_capacity=[1, 1, 1, 1]),
        pd.concat(
            [
                retrieved("DE_LU", "FR", offered_capacity=[1, 1, 1, 1]),
                retrieved("FR", "DE_LU", scheduled_exchange=[np.nan] * 4),
            ]
        ),
        pd.concat(
            [
                retrieved("DE_LU", "FR", offered_capacity=[1, 1, 1, 1]),
                retrieved("FR", "DE_LU", ntc=[1, np.nan, np.nan, 1]),
            ]
        ),
    ],
    ids=["missing-direction", "missing-data", "long-gap"],
)
def test_build_capacities_raises(network, data):
    pairs = border_shares(inter_zone_branches(network, AREA_TO_BUS))
    with pytest.raises(AssertionError, match="FR->DE_LU"):
        build_capacities(
            data, pairs, SNAPSHOTS, {}, max_gap=1, exchange_proxy_borders=[]
        )


@pytest.mark.parametrize(
    "value, expected",
    [(800.0, [600, 200]), ("physical", [3000, 1000]), (0.0, [0, 0])],
    ids=["mw", "physical", "zero"],
)
def test_build_capacities_fixed_borders(network, value, expected):
    pairs = border_shares(inter_zone_branches(network, AREA_TO_BUS))
    empty = retrieved("ES", "FR").iloc[:0]

    capacities, sources = build_capacities(
        empty,
        pairs,
        SNAPSHOTS,
        {"DE_LU-FR": value},
        max_gap=1,
        exchange_proxy_borders=[],
    )

    for bus in ["DE00", "LUG1"]:
        forward, reverse = f"{bus}->FR00", f"FR00->{bus}"
        np.testing.assert_allclose(capacities[forward], capacities[reverse])
    np.testing.assert_allclose(capacities["DE00->FR00"], expected[0])
    np.testing.assert_allclose(capacities["LUG1->FR00"], expected[1])
    assert (sources.source == "fixed").all()


@pytest.mark.parametrize(
    "forward, reverse, p_nom",
    [
        ([100, 200, 300, 400], [50, 50, 50, 50], 400),
        ([10, 10, 10, 10], [80, 40, 20, 0], 80),
    ],
    ids=["forward-max", "reverse-max"],
)
def test_apply_cross_border_capacities(network, forward, reverse, p_nom):
    capacities = pd.DataFrame(
        {
            "DE00->FR00": forward,
            "FR00->DE00": reverse,
            "FR00->LUG1": [0.0] * 4,
            "LUG1->FR00": [0.0] * 4,
        },
        index=SNAPSHOTS,
        dtype=float,
    )

    apply_cross_border_capacities(network, capacities, AREA_TO_BUS)

    assert network.lines.empty
    assert set(network.links.index) == {
        "copper_DE00_LUG1_0",
        "DE00->FR00",
        "FR00->LUG1",
    }
    link = network.links.loc["DE00->FR00"]
    assert (link.bus0, link.bus1, link.carrier) == ("DE00", "FR00", "DC")
    assert link.p_nom == p_nom
    assert link.length == 400.0
    np.testing.assert_allclose(
        network.links_t.p_max_pu["DE00->FR00"], np.array(forward) / p_nom
    )
    np.testing.assert_allclose(
        network.links_t.p_min_pu["DE00->FR00"], -np.array(reverse) / p_nom
    )
    np.testing.assert_allclose(network.links_t.p_max_pu["FR00->LUG1"], 0.0)


def test_apply_cross_border_capacities_raises_on_missing_border(network):
    capacities = pd.DataFrame(
        1.0, index=SNAPSHOTS, columns=["DE00->FR00", "FR00->DE00"]
    )
    with pytest.raises(AssertionError, match="LUG1"):
        apply_cross_border_capacities(network, capacities, AREA_TO_BUS)


@pytest.mark.parametrize(
    "sources, areas, expected",
    [
        (["scheduled_exchange"], AREAS, {"DE00->FR00": 1500.0, "FR00->DE00": 1500.0}),
        (["scheduled_exchange"], ["FR"], {"DE00->FR00": 100.0, "FR00->DE00": 50.0}),
        (["ntc"], AREAS, {"LUG1->FR00": 500.0, "FR00->LUG1": 500.0}),
    ],
    ids=["raised", "one-end", "other-source"],
)
def test_apply_cross_border_capacities_hybrid(network, sources, areas, expected):
    network.lines.s_max_pu = 0.7
    capacities = pd.DataFrame(
        {"DE00->FR00": 100.0, "FR00->DE00": 50.0, "FR00->LUG1": 0.0, "LUG1->FR00": 0.0},
        index=SNAPSHOTS,
    )
    border_sources = pd.DataFrame(
        {
            "area_from": ["DE_LU", "FR", "FR", "DE_LU"],
            "area_to": ["FR", "DE_LU", "DE_LU", "FR"],
            "source": ["scheduled_exchange"] * 2 + ["ntc"] * 2,
        },
        index=capacities.columns,
    )
    raised = raise_to_physical(
        network, capacities, AREA_TO_BUS, border_sources, 0.5, 0.5, sources, areas
    )
    apply_cross_border_capacities(network, raised, AREA_TO_BUS)

    links = network.links
    p_max = network.links_t.p_max_pu.mul(links.p_nom).iloc[0]
    p_min = network.links_t.p_min_pu.mul(links.p_nom).iloc[0]
    result = {
        "DE00->FR00": p_max["DE00->FR00"],
        "FR00->DE00": -p_min["DE00->FR00"],
        "FR00->LUG1": p_max["FR00->LUG1"],
        "LUG1->FR00": -p_min["FR00->LUG1"],
    }
    for name, value in expected.items():
        assert result[name] == pytest.approx(value)


def test_raise_to_physical_fails_on_unknown_area(network):
    with pytest.raises(AssertionError, match="Unknown ENTSO-E areas"):
        raise_to_physical(
            network,
            pd.DataFrame(),
            AREA_TO_BUS,
            pd.DataFrame(),
            0.5,
            0.5,
            ["ntc"],
            ["XX"],
        )


@pytest.mark.parametrize(
    "line_factor, dc_factor, expected",
    [(0.49, 0.7, 0.49 * 3000 + 0.7 * 1000), (0.7, 0.7, 0.7 * 4000)],
    ids=["separate", "uniform"],
)
def test_raise_to_physical_weights_lines_and_dc_links(
    network, line_factor, dc_factor, expected
):
    network.add(
        "Link", "DE-FR-DC", bus0="DE00", bus1="FR00", carrier="DC", p_nom=1000.0
    )
    names = ["DE00->FR00", "FR00->DE00"]
    capacities = pd.DataFrame(0.0, index=SNAPSHOTS, columns=names)
    border_sources = pd.DataFrame(
        {"area_from": ["DE_LU", "FR"], "area_to": ["FR", "DE_LU"]}, index=names
    ).assign(source="scheduled_exchange")
    raised = raise_to_physical(
        network,
        capacities,
        AREA_TO_BUS,
        border_sources,
        line_factor,
        dc_factor,
        ["scheduled_exchange"],
        AREAS,
    )
    np.testing.assert_allclose(raised.values, expected)


@pytest.mark.parametrize(
    "exchange, expected",
    [
        ([1.0, 2, 3, 4], [1.0, 2, 3, 4]),
        ([np.nan, 2, np.nan, 4], [np.nan, 2, np.nan, 4]),
    ],
)
def test_scheduled_exchanges(exchange, expected):
    data = pd.concat(
        [
            retrieved("DE_LU", "FR", scheduled_exchange=exchange),
            retrieved("FR", "DE_LU"),
        ]
    )
    result = scheduled_exchanges(
        data, SNAPSHOTS.append(SNAPSHOTS[-1:] + pd.Timedelta("1h"))
    )

    assert result.columns.tolist() == ["DE_LU->FR", "FR->DE_LU"]
    np.testing.assert_array_equal(result["DE_LU->FR"], [*expected, np.nan])
    np.testing.assert_array_equal(result["FR->DE_LU"], [0.0, 0, 0, 0, np.nan])


EXTERNAL_BORDERS = pd.DataFrame(
    {
        "area": ["DE_LU", "FR"],
        "external_area": ["RU", "UA"],
        "flow": ["scheduled", "physical"],
    }
)


def external(area_from: str, area_to: str, total: list[float]) -> pd.DataFrame:
    return retrieved(area_from, area_to).assign(
        scheduled_exchange_total=total, physical_flow=[5.0] * 4
    )


@pytest.mark.parametrize(
    "directions, net, warned",
    [
        (
            {
                ("RU", "DE_LU"): [100.0, 0, 50, np.nan],
                ("DE_LU", "RU"): [0.0, 30, 0, np.nan],
            },
            [100.0, -30, 50, 0],
            True,
        ),
        ({("RU", "DE_LU"): [1.0, 2, 3, 4]}, [1.0, 2, 3, 4], False),
        ({("DE_LU", "RU"): [1.0, np.nan, 3, 4]}, [-1.0, 0, -3, -4], True),
    ],
)
def test_external_net_imports(directions, net, warned, caplog):
    data = pd.concat(
        [external("FR", "UA", [np.nan] * 4)]
        + [external(*d, values) for d, values in directions.items()]
    )
    result = external_net_imports(
        data, EXTERNAL_BORDERS, SNAPSHOTS, "scheduled_exchange_total"
    )

    assert result.columns.tolist() == ["RU->DE_LU", "UA->FR"]
    np.testing.assert_array_equal(result["RU->DE_LU"], net)
    np.testing.assert_array_equal(result["UA->FR"], [-5.0] * 4)
    assert ("RU->DE_LU" in caplog.text) == warned


@pytest.mark.parametrize(
    "data, match",
    [
        (retrieved("RU", "DE_LU"), "scheduled_exchange_total"),
        (external("FR", "UA", [1.0] * 4), "RU->DE_LU"),
    ],
)
def test_external_net_imports_fails(data, match):
    with pytest.raises(AssertionError, match=match):
        external_net_imports(
            data, EXTERNAL_BORDERS, SNAPSHOTS, "scheduled_exchange_total"
        )


def test_apply_external_flows_fails_on_unknown_area(tmp_path):
    borders = tmp_path / "borders.csv"
    EXTERNAL_BORDERS.to_csv(borders, index=False)
    area_to_bus = tmp_path / "area_to_bus.csv"
    AREA_TO_BUS.rename_axis("bus").rename("entsoe_area").to_csv(area_to_bus)
    inputs = SimpleNamespace(area_to_bus=area_to_bus, external_borders=borders)
    with pytest.raises(AssertionError, match="MA"):
        apply_external_flows(
            pypsa.Network(), inputs, ["UA", "MA"], "scheduled_exchange_total"
        )


def test_add_external_flows(network):
    network.add(
        "Load",
        ["DE00", "LUG1", "FR00"],
        bus=["DE00", "LUG1", "FR00"],
        p_set=pd.DataFrame({"DE00": 10.0, "LUG1": 1.0, "FR00": 5.0}, index=SNAPSHOTS),
    )
    net_imports = pd.DataFrame(
        {"RU->DE_LU": [100.0, -30, 50, 0], "UA->FR": [-5.0] * 4}, index=SNAPSHOTS
    )

    add_external_flows(network, net_imports, EXTERNAL_BORDERS, AREA_TO_BUS)

    imported = network.get_switchable_as_dense("Generator", "p_max_pu").mul(
        network.generators.p_nom
    )
    p_min_pu = network.get_switchable_as_dense("Generator", "p_min_pu")
    exported = network.loads_t.p_set[["RU->DE_LU export", "UA->FR export"]]
    assert network.generators.bus.to_dict() == {"RU->DE_LU import": "DE00"}
    assert network.loads.loc[exported.columns, "bus"].tolist() == ["DE00", "FR00"]
    pd.testing.assert_frame_equal(
        p_min_pu, network.get_switchable_as_dense("Generator", "p_max_pu")
    )
    np.testing.assert_allclose(
        imported.sum(axis=1) - exported.sum(axis=1), net_imports.sum(axis=1)
    )


def test_add_external_flows_fails_on_modelled_area(network):
    borders = pd.DataFrame({"area": ["DE_LU"], "external_area": ["FR"]})
    net_imports = pd.DataFrame({"FR->DE_LU": [1.0] * 4}, index=SNAPSHOTS)
    with pytest.raises(AssertionError, match="FR"):
        add_external_flows(network, net_imports, borders, AREA_TO_BUS)
