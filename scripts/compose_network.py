# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Compose the network for one planning horizon from the electricity and
sector-coupling building blocks.

Starting from the clustered base network, the script adds the electricity
components (`add_electricity`), the sector-coupling components if enabled
(`prepare_sector_network`), applies temporal aggregation and the final
preparation for solving (`prepare_network`), and then handles the planning
horizon according to the foresight mode. In the first horizon of a myopic or
perfect foresight run, existing capacities are added (`add_existing_baseyear`).
In later myopic horizons, the optimised capacities of the previous horizon's
solved network are carried over as brownfield (`add_brownfield`). Under
perfect foresight, the horizon is stacked onto the previous composed network
to form a multi-period network (`prepare_perfect_foresight`). Finally, CO2
budget constraints and configured adjustments to costs and potentials are
applied and the network is exported for solving.
"""

import logging

import pandas as pd
import pypsa
from snakemake.iocontainers import InputFiles

from scripts._helpers import (
    configure_logging,
    load_costs,
    sanitize_custom_columns,
    set_scenario_config,
)
from scripts.add_brownfield import adjust_renewable_capacity_limits
from scripts.add_brownfield import main as apply_brownfield
from scripts.add_electricity import main as add_electricity_components
from scripts.add_electricity import (
    sanitize_carriers,
    sanitize_locations,
)
from scripts.add_existing_baseyear import main as add_existing_capacities
from scripts.build_cross_border_capacities import read_retrieved
from scripts.lib.backcast_calibration import (
    apply_monthly_availability,
    calibrate_solar,
    calibrate_wind,
    read_generation,
    read_solar_unpublished,
)
from scripts.lib.backcast_dispatch import dispatch_carriers
from scripts.lib.conventional import (
    apply_chp_floor,
    apply_energy_budget,
    apply_nuclear_min_output,
    clip_p_min_pu,
    set_fuel_price,
    spread_marginal_cost,
)
from scripts.lib.cross_border import (
    add_external_flows,
    apply_cross_border_capacities,
    external_net_imports,
    raise_to_physical,
)
from scripts.lib.entsoe import (
    assert_known_areas,
    read_area_to_bus,
    read_hourly_generation,
)
from scripts.lib.eurostat import eurostat_year, gross_electricity
from scripts.lib.unavailability import apply_unavailability
from scripts.prepare_network import (
    apply_co2_budget_constraints,
    apply_temporal_aggregation,
    maybe_adjust_costs_and_potentials,
)
from scripts.prepare_network import main as prepare_network_for_solving
from scripts.prepare_perfect_foresight import apply_phase_outs
from scripts.prepare_perfect_foresight import main as prepare_perfect_foresight
from scripts.prepare_sector_network import (
    main as add_sector_components,
)

logger = logging.getLogger(__name__)

SECTOR_RENEWABLE_CARRIERS = {"solar rooftop"}


def apply_backcast(
    n: pypsa.Network,
    inputs: InputFiles,
    area_to_bus: pd.Series | None,
    electricity: dict,
    backcast: dict,
) -> None:
    """Apply the ENTSO-E cross-border capacities and the backcast calibration."""
    ntc = electricity["cross_border_capacities"] == "ntc"
    wind = backcast["renewables"]["wind"]
    solar = backcast["renewables"]["solar"]
    nuclear = backcast["dispatch"]["nuclear"]
    dispatch = dispatch_carriers(backcast["dispatch"])
    monthly = dispatch.uses_monthly_generation
    if not (
        ntc or wind["calibrate"] or solar["calibrate"] or monthly or dispatch.outages
    ):
        return
    assert area_to_bus is not None
    if ntc:
        capacities = pd.read_csv(
            inputs.cross_border_capacities, index_col=0, parse_dates=True
        )
        hybrid = backcast["interconnection"]["hybrid"]
        if hybrid["enable"]:
            capacities = raise_to_physical(
                n,
                capacities,
                area_to_bus,
                pd.read_csv(inputs.cross_border_capacities_sources, index_col=0),
                hybrid["physical_factor"],
                hybrid["dc_physical_factor"],
                hybrid["sources"],
                hybrid["areas"],
            )
        apply_cross_border_capacities(n, capacities, area_to_bus)
    if wind["calibrate"] or solar["calibrate"] or monthly:
        generation = read_generation(inputs.entsoe_generation)
    if wind["calibrate"]:
        calibrate_wind(
            n,
            generation,
            area_to_bus,
            wind["curtailment_twh"],
            wind["exclude"],
            wind["factor_bounds"],
            wind["target_twh"],
        )
    if solar["calibrate"]:
        calibrate_solar(
            n,
            generation,
            area_to_bus,
            solar["exclude"],
            solar["factor_bounds"],
            pd.read_csv(inputs.solar_self_consumption, index_col="country").share,
            (
                read_solar_unpublished(
                    inputs.solar_unpublished, eurostat_year(n.snapshots, None)
                )
                if solar["unpublished"]
                else pd.Series(dtype=float)
            ),
        )
    if dispatch.nuclear_monthly:
        apply_monthly_availability(
            n,
            generation,
            area_to_bus,
            "nuclear",
            nuclear["area_map"],
            nuclear["phase_out"],
            exclude=dispatch.exclude["nuclear"],
        )
    for carrier in dispatch.monthly:
        apply_monthly_availability(
            n,
            generation,
            area_to_bus,
            carrier,
            {},
            {},
            keep_unpublished=True,
            exclude=dispatch.exclude[carrier],
        )
    if dispatch.outages:
        share = pd.read_csv(
            inputs.unavailability, index_col=0, header=[0, 1], parse_dates=True
        )
        coverage = pd.read_csv(inputs.unavailability_coverage, index_col=[0, 1])
        apply_unavailability(
            n,
            share,
            coverage,
            area_to_bus,
            dispatch.min_coverage,
            nuclear["phase_out"],
        )


def apply_external_flows(
    n: pypsa.Network,
    inputs: InputFiles,
    area_to_bus: pd.Series,
    areas: list[str],
    source: str,
) -> None:
    """Add the hourly exchanges with the external `areas` as fixed flows at their border buses."""
    borders = pd.read_csv(inputs.external_borders)
    unknown = sorted(set(areas) - set(borders.external_area))
    assert not unknown, f"backcast: areas: external: areas without border: {unknown}"
    modelled = set(area_to_bus[area_to_bus.index.isin(n.buses.index)])
    borders = borders[borders.external_area.isin(areas) & borders.area.isin(modelled)]
    data = read_retrieved(inputs.external_exchanges)
    net_imports = external_net_imports(data, borders, n.snapshots, source)
    add_external_flows(n, net_imports, borders, area_to_bus)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "compose_network",
            run="test-run",
            horizon="2050",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Extract configuration and parameters
    config = snakemake.config
    params = snakemake.params
    inputs = snakemake.input

    current_horizon = int(snakemake.wildcards.horizon)
    horizons = params.horizons
    foresight = params.foresight
    sector_mode = params.sector["enabled"]

    is_first_horizon = current_horizon == horizons[0]

    phase_outs = params.existing_capacities["phase_outs"]
    if phase_outs and foresight != "perfect":
        logger.warning(
            "existing_capacities.phase_outs is only applied under perfect foresight; "
            f"the {len(phase_outs)} configured rule(s) are ignored for foresight "
            f"'{foresight}', where phase-outs follow the powerplant lifetimes instead."
        )

    n = pypsa.Network(snakemake.input.network)
    n_previous = None if is_first_horizon else pypsa.Network(inputs.network_previous)
    nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(snakemake.input.tech_costs)
    dispatch = dispatch_carriers(params.backcast["dispatch"])
    set_fuel_price(costs, dispatch.fuel_price)

    add_electricity_components(n, inputs, params, costs)

    area_to_bus = read_area_to_bus(inputs.area_to_bus) if inputs.area_to_bus else None
    apply_backcast(n, inputs, area_to_bus, params.electricity, params.backcast)

    external_flows = params.backcast["areas"]["external"]
    if external_flows["enable"]:
        assert area_to_bus is not None
        apply_external_flows(
            n, inputs, area_to_bus, external_flows["areas"], external_flows["source"]
        )

    if sector_mode:
        add_sector_components(n, inputs, params, costs, nyears, current_horizon)

    apply_temporal_aggregation(n, inputs, params)

    renewable_carriers = sorted(
        set(params.renewable_carriers)
        | (SECTOR_RENEWABLE_CARRIERS & set(n.generators.carrier))
    )

    if foresight != "overnight" and is_first_horizon:
        add_existing_capacities(n, inputs, params, costs)
        if foresight == "myopic":
            adjust_renewable_capacity_limits(
                n, str(current_horizon), renewable_carriers
            )

    prepare_network_for_solving(n, inputs, params, costs, nyears, current_horizon)

    if foresight == "myopic" and not is_first_horizon:
        apply_brownfield(
            n, n_previous, inputs, params, current_horizon, renewable_carriers
        )

    if foresight == "perfect":
        n = prepare_perfect_foresight(n, n_previous, params, current_horizon)
        apply_phase_outs(n, phase_outs, horizons)

    apply_co2_budget_constraints(
        n, inputs=inputs, params=params, nyears=nyears, current_horizon=current_horizon
    )

    adjustments = params.adjustments
    maybe_adjust_costs_and_potentials(n, adjustments["electricity"], current_horizon)
    maybe_adjust_costs_and_potentials(n, adjustments["sector"], current_horizon)

    chp_floor = params.backcast["dispatch"]["chp_floor"]
    if chp_floor["enable"]:
        floor = pd.read_csv(
            inputs.chp_floor, index_col=0, header=[0, 1], parse_dates=True
        )
        apply_chp_floor(n, floor, chp_floor["replace_p_min_pu"])

    nuclear = params.backcast["dispatch"]["nuclear"]
    spread = params.conventional["marginal_cost_spread"]
    observed_spread = spread["enable"] and spread["reference"] == "observed"
    if observed_spread or dispatch.uses_hourly_generation:
        assert area_to_bus is not None
        generation = read_hourly_generation(inputs.entsoe_generation_hourly)
        if dispatch.nuclear_min_output:
            apply_nuclear_min_output(
                n,
                generation,
                area_to_bus,
                nuclear["area_map"],
                nuclear["load_following"],
            )
        for carrier in dispatch.budget:
            reported = (
                gross_electricity(
                    inputs.eurostat_production,
                    eurostat_year(n.snapshots, None),
                    carrier,
                )
                if carrier in dispatch.eurostat
                else pd.Series(dtype=float)
            )
            apply_energy_budget(
                n,
                generation,
                area_to_bus,
                carrier,
                dispatch.exclude[carrier],
                dispatch.monthly_ceiling[carrier],
                reported,
                carrier in dispatch.fixed,
            )

    clip_p_min_pu(n)
    if spread["enable"]:
        assert foresight == "overnight", (
            "conventional.marginal_cost_spread requires overnight foresight, got "
            f"'{foresight}'."
        )
        observed = (None, None)
        if observed_spread:
            assert area_to_bus is not None
            exclude = spread["exclude_areas"]
            assert_known_areas(
                exclude, area_to_bus, "conventional.marginal_cost_spread.exclude_areas"
            )
            observed = (
                generation.drop(columns=exclude, level="area", errors="ignore"),
                area_to_bus,
            )
        spread_marginal_cost(
            n, spread["carriers"], spread["spread"], spread["utilisation"], *observed
        )

    sanitize_custom_columns(n)
    sanitize_carriers(n, config)
    sanitize_locations(n)
    n.meta = dict(config, **dict(wildcards=dict(snakemake.wildcards)))
    if n.investment_periods.empty:
        n.consistency_check()
    else:
        logger.warning(
            "Skipping consistency check for multi-period (perfect foresight) network: "
            "known pandas/PyPSA MultiIndex limitation."
        )

    logger.info(f"Exporting composed network for horizon {current_horizon}")
    n.export_to_netcdf(snakemake.output[0])
