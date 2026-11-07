# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calibration of generator availabilities to the actual generation reported by the ENTSO-E Transparency Platform.
"""

import logging

import pandas as pd
import pypsa

from scripts.lib.entsoe import assert_known_areas

logger = logging.getLogger(__name__)

WIND_CARRIERS = ["onwind", "offwind"]
CLIP_TOLERANCE = 0.01


def read_generation(path: str) -> pd.DataFrame:
    """Read the monthly generation in MWh and hours with data indexed by area, carrier and month."""
    generation = pd.read_csv(path, parse_dates=["month"])
    return generation.set_index(["area", "carrier", "month"])


def snapshot_months(n: pypsa.Network) -> pd.DatetimeIndex:
    """Return the first day of the month of each snapshot."""
    return n.snapshots.to_period("M").to_timestamp()


def month_hours(n: pypsa.Network) -> pd.Series:
    """Return the weighted hours of each snapshot month."""
    return n.snapshot_weightings.generators.groupby(snapshot_months(n)).sum()


def carrier_generation(generation: pd.DataFrame, carriers: list[str]) -> pd.Series:
    """Return the mean generation in MW over the hours with data of `carriers` per area, carrier and month."""
    selected = generation[generation.index.get_level_values("carrier").isin(carriers)]
    return selected.generation / selected.hours


def assert_mapped(generators: pd.DataFrame, area_to_bus: pd.Series) -> None:
    """Assert that all `generators` are on buses with an ENTSO-E area."""
    unmapped = generators.index[~generators.bus.isin(area_to_bus.index)]
    assert unmapped.empty, (
        f"Calibrated generators on buses without ENTSO-E area: {unmapped.tolist()}"
    )


def check_clipping(label: str, target: float, clipped: float) -> None:
    """Log the energy after clipping and warn if it deviates from the target by more than ``CLIP_TOLERANCE``."""
    message = f"{label}: target {target / 1e6:.2f} TWh, after clipping {clipped / 1e6:.2f} TWh."
    logger.info(message)
    if abs(target - clipped) > CLIP_TOLERANCE * target:
        logger.warning(f"Clipping loses more than {CLIP_TOLERANCE:.0%} of {message}")


def calibrate_wind(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    curtailment_twh: dict[str, float | dict[str, float]],
    exclude: dict[str, list[str]],
    factor_bounds: list[float],
    target_twh: dict[str, dict[str, float]],
) -> None:
    """
    Scale the onshore and offshore wind availability per area to the actual generation.

    Only the months with actual generation of a carrier are compared. Their
    actual energy is the generation divided by the hours with data times the
    hours of the month. The target energy adds the curtailment of the area,
    given per carrier or as a total split by the actual generation. The
    available energy of the model is taken over the same months. Areas without
    published generation are scaled to the energy in `target_twh` over all
    snapshots instead. All generators of a carrier in the area get the same
    factor for all snapshots, ``p_max_pu`` is clipped at 1. Generators of a
    carrier without actual generation in an area that publishes generation are
    set to zero availability. Generators of areas without published generation
    and without target or excluded in `exclude` keep their availability.
    Factors outside `factor_bounds` raise an error.

    Parameters
    ----------
    n : pypsa.Network
        Network with time-varying ``p_max_pu`` of the wind generators.
    generation : pd.DataFrame
        Monthly generation in MWh and hours with data indexed by area, carrier and month.
    area_to_bus : pd.Series
        ENTSO-E area per bus.
    curtailment_twh : dict[str, float | dict[str, float]]
        Curtailed wind energy in TWh per area added to the actual generation,
        in total or per wind carrier.
    exclude : dict[str, list[str]]
        Wind carriers per area whose availability is not calibrated.
    factor_bounds : list[float]
        Lower and upper bound of the calibration factors.
    target_twh : dict[str, dict[str, float]]
        Wind energy in TWh per carrier over all snapshots for areas without
        published generation.
    """
    assert_known_areas(
        [*curtailment_twh, *exclude, *target_twh], area_to_bus, "the wind calibration"
    )
    ignored = sorted(set(curtailment_twh) & set(exclude))
    assert not ignored, (
        f"Wind curtailment given for areas with excluded carriers: {ignored}"
    )
    published = generation.index.unique("area")
    targeted_published = sorted(set(target_twh) & set(published))
    assert not targeted_published, (
        f"Wind targets given for areas with published generation: {targeted_published}"
    )
    overlap = sorted(
        (area, carrier)
        for area, targets in target_twh.items()
        for carrier in set(targets) & set(exclude.get(area, []))
    )
    assert not overlap, f"Wind targets given for excluded carriers: {overlap}"
    g = n.generators
    wind = g.index[g.carrier.str.startswith(tuple(WIND_CARRIERS))]
    assert_mapped(g.loc[wind], area_to_bus)
    p_max_pu = n.generators_t.p_max_pu
    static = wind.difference(p_max_pu.columns)
    assert static.empty, (
        f"Wind generators without time-varying p_max_pu: {static.tolist()}"
    )
    gen_areas = g.bus.map(area_to_bus)
    weights = n.snapshot_weightings.generators
    months = snapshot_months(n)
    power = carrier_generation(generation, WIND_CARRIERS)
    energy = (
        power * month_hours(n).reindex(power.index.get_level_values("month")).values
    )
    lower, upper = factor_bounds

    def available(p_max_pu: pd.DataFrame, mask: pd.Series) -> float:
        p_nom = g.p_nom[p_max_pu.columns]
        return p_max_pu.loc[mask].mul(p_nom).mul(weights[mask], axis=0).sum().sum()

    for area in area_to_bus.unique():
        area_energy = (
            energy[area].groupby(level="carrier").sum()
            if area in energy.index.unique("area")
            else pd.Series(dtype=float)
        )
        value = curtailment_twh.get(area, 0.0)
        covered = (
            set(value) <= set(area_energy.index)
            if isinstance(value, dict)
            else not value or not area_energy.empty
        )
        assert covered, (
            f"Wind curtailment given for {area} without actual generation of its wind carriers."
        )
        curtailment = carrier_curtailment(value, area_energy)
        for carrier in WIND_CARRIERS:
            gens = g.index[gen_areas.eq(area) & g.carrier.str.startswith(carrier)]
            capacity = g.p_nom[gens].sum()
            if carrier in exclude.get(area, []):
                logger.info(f"Excluded {carrier} in {area} from the wind calibration.")
                continue
            if carrier in target_twh.get(area, {}):
                mask = pd.Series(True, n.snapshots)
                target = target_twh[area][carrier] * 1e6
            elif carrier in area_energy.index:
                mask = months.isin(energy[area][carrier].index)
                target = area_energy[carrier] + curtailment[carrier]
            elif area in published and capacity > 0:
                p_max_pu[gens] = 0.0
                logger.warning(
                    f"No actual {carrier} generation published for {area}, setting "
                    f"the availability of its {capacity:.0f} MW to zero."
                )
                continue
            else:
                if capacity > 0:
                    logger.info(
                        f"No actual {carrier} generation for {area}, keeping its availability."
                    )
                continue
            before = available(p_max_pu[gens], mask)
            assert before > 0, (
                f"Wind target or actual {carrier} generation in {area} without {carrier} generators."
            )
            factor = target / before
            assert lower <= factor <= upper, (
                f"Wind calibration factor {factor:.2f} of {carrier} in {area} is outside "
                f"the bounds [{lower}, {upper}]. Add {carrier} of {area} to backcast: "
                "wind_calibration_exclude or widen backcast: "
                "wind_calibration_factor_bounds."
            )
            p_max_pu[gens] = (p_max_pu[gens] * factor).clip(upper=1)
            check_clipping(
                f"Calibrated {carrier} in {area} over {mask.sum()} snapshots "
                f"from {before / 1e6:.2f} TWh (x{factor:.2f})",
                target,
                available(p_max_pu[gens], mask),
            )


def carrier_curtailment(
    curtailment: float | dict[str, float], energy: pd.Series
) -> pd.Series:
    """Return the curtailment in MWh per wind carrier, splitting a total by the actual generation."""
    if isinstance(curtailment, dict):
        return (
            pd.Series(curtailment, dtype=float).reindex(energy.index, fill_value=0.0)
            * 1e6
        )
    if not curtailment:
        return energy * 0.0
    return curtailment * 1e6 * energy / energy.sum()


def apply_monthly_availability(
    n: pypsa.Network,
    generation: pd.DataFrame,
    area_to_bus: pd.Series,
    carrier: str,
    area_map: dict[str, str],
    phase_out: dict[str, str],
    keep_unpublished: bool = False,
    exclude: list[str] | None = None,
) -> None:
    """
    Set the availability of a carrier per area and month to its actual capacity factor.

    The capacity factor is the monthly generation divided by the hours with
    data and the capacity of the carrier in the area, clipped to [0, 1]. After
    the last operating date of an area in `phase_out`, the availability is
    zero. Every month with snapshots before it must have actual generation.
    Generators of areas without any published generation keep their
    availability. Generators of the keys of `area_map` are pooled with the
    area they map to. With `keep_unpublished`, generators of areas that
    publish generation, but none of the carrier, also keep their availability
    instead of raising an error. Generators of areas in `exclude` keep their
    availability regardless of published generation, e.g. a flexible plant
    fleet whose monthly mean capacity factor would wrongly flatten its hourly
    dispatch, such as a carrier used as a swing exporter.

    Parameters
    ----------
    n : pypsa.Network
        Network with snapshots.
    generation : pd.DataFrame
        Monthly generation in MWh and hours with data indexed by area, carrier and month.
    area_to_bus : pd.Series
        ENTSO-E area per bus.
    carrier : str
        Generator carrier, e.g. ``nuclear``.
    area_map : dict[str, str]
        Area whose generators are calibrated with the actual generation of another area.
    phase_out : dict[str, str]
        Last operating date of the carrier per area as ISO date.
    keep_unpublished : bool
        Keep the availability of the carrier in areas that publish generation,
        but none of the carrier.
    exclude : list[str] | None
        ENTSO-E areas kept at their availability regardless of published
        generation.
    """
    exclude = exclude or []
    assert_known_areas(
        [*area_map, *area_map.values(), *phase_out, *exclude],
        area_to_bus,
        f"the monthly {carrier} availability",
    )
    g = n.generators.query("carrier == @carrier")
    assert_mapped(g, area_to_bus)
    areas = g.bus.map(area_to_bus).replace(area_map)
    published = areas.isin(generation.index.unique("area")) & ~areas.isin(exclude)
    if not areas.isin(generation.index.unique("area")).all():
        logger.info(
            f"No actual generation published for "
            f"{sorted(areas[~areas.isin(generation.index.unique('area'))].unique())}, "
            f"keeping their {carrier} availability."
        )
    if exclude:
        excluded = areas.isin(exclude)
        if excluded.any():
            logger.info(
                f"Excluded {sorted(areas[excluded].unique())} from the monthly "
                f"{carrier} availability, keeping their availability."
            )
    if keep_unpublished:
        is_carrier = generation.index.get_level_values("carrier") == carrier
        carrier_areas = generation.index[is_carrier].unique("area")
        unpublished = published & ~areas.isin(carrier_areas)
        if unpublished.any():
            logger.info(
                f"No actual {carrier} generation published for "
                f"{sorted(areas[unpublished].unique())}, keeping their {carrier} availability."
            )
        published &= ~unpublished
    areas = areas[published]
    capacity = g.p_nom[areas.index].groupby(areas).sum()
    ignored = sorted(set(phase_out) - set(capacity.index))
    assert not ignored, (
        f"{carrier.capitalize()} phase-out given for areas without calibrated "
        f"{carrier} generators: {ignored}"
    )
    power = carrier_generation(generation, [carrier]).droplevel("carrier")
    power = power.unstack("area").reindex(columns=capacity.index)
    months = snapshot_months(n)
    end = pd.Series(
        {
            area: pd.Timestamp(date) + pd.Timedelta(days=1)
            for area, date in phase_out.items()
        },
        dtype="datetime64[ns]",
    ).reindex(capacity.index, fill_value=pd.Timestamp.max)
    operating = pd.DataFrame(
        n.snapshots.values[:, None] < end.values,
        index=n.snapshots,
        columns=capacity.index,
    )
    for area in capacity.index:
        required = months[operating[area].values].unique()
        missing = required.difference(power[area].dropna().index)
        assert missing.empty, (
            f"No actual {carrier} generation for {area} in {missing.strftime('%Y-%m').tolist()} "
            "before its phase-out. Set its last operating date in backcast: nuclear_phase_out."
        )
    target = power.div(capacity).reindex(months).set_axis(n.snapshots)
    target = target.where(operating, 0.0)
    capacity_factor = target.clip(0, 1)
    weights = n.snapshot_weightings.generators
    for area in capacity.index:
        check_clipping(
            f"Monthly {carrier} availability in {area} with mean {capacity_factor[area].mean():.2f}",
            target[area].mul(weights).sum() * capacity[area],
            capacity_factor[area].mul(weights).sum() * capacity[area],
        )
    n.generators_t.p_max_pu[areas.index] = capacity_factor[areas].set_axis(
        areas.index, axis=1
    )
