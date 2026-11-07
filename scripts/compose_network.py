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
from scripts.lib.backcast_calibration import (
    apply_monthly_availability,
    calibrate_wind,
    read_generation,
)
from scripts.lib.conventional import clip_p_min_pu, spread_marginal_cost
from scripts.lib.cross_border import apply_cross_border_capacities, raise_to_physical
from scripts.lib.entsoe import assert_known_areas, read_area_to_bus
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
    n: pypsa.Network, inputs: InputFiles, electricity: dict, backcast: dict
) -> None:
    """Apply the ENTSO-E cross-border capacities and the backcast calibration."""
    ntc = electricity["cross_border_capacities"] == "ntc"
    wind = backcast["calibrate_wind"]
    nuclear = backcast["monthly_nuclear_availability"]
    carriers = backcast["monthly_availability_carriers"]
    if not (ntc or wind or nuclear or carriers):
        return
    area_to_bus = read_area_to_bus(inputs.area_to_bus)
    if ntc:
        capacities = pd.read_csv(
            inputs.cross_border_capacities, index_col=0, parse_dates=True
        )
        hybrid = backcast["hybrid_capacities"]
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
    if wind or nuclear or carriers:
        generation = read_generation(inputs.entsoe_generation)
    if wind:
        calibrate_wind(
            n,
            generation,
            area_to_bus,
            backcast["wind_curtailment_twh"],
            backcast["wind_calibration_exclude"],
            backcast["wind_calibration_factor_bounds"],
            backcast["wind_target_twh"],
        )
    if nuclear:
        apply_monthly_availability(
            n,
            generation,
            area_to_bus,
            "nuclear",
            backcast["nuclear_area_map"],
            backcast["nuclear_phase_out"],
        )
    for carrier in carriers:
        apply_monthly_availability(
            n,
            generation,
            area_to_bus,
            carrier,
            {},
            {},
            keep_unpublished=True,
            exclude=backcast["monthly_availability_exclude"].get(carrier, []),
        )


def read_observed_generation(
    inputs: InputFiles, exclude_areas: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Read the hourly ENTSO-E generation per area and carrier without `exclude_areas` and the area-to-bus map."""
    area_to_bus = read_area_to_bus(inputs.area_to_bus)
    assert_known_areas(
        exclude_areas, area_to_bus, "conventional.marginal_cost_spread.exclude_areas"
    )
    generation = pd.read_csv(
        inputs.entsoe_generation_hourly, index_col=0, header=[0, 1], parse_dates=True
    ).drop(columns=exclude_areas, level="area", errors="ignore")
    return generation, area_to_bus


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

    add_electricity_components(n, inputs, params, costs)

    apply_backcast(n, inputs, params.electricity, params.backcast)

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

    clip_p_min_pu(n)
    spread = params.conventional["marginal_cost_spread"]
    if spread["enable"]:
        assert foresight == "overnight", (
            "conventional.marginal_cost_spread requires overnight foresight, got "
            f"'{foresight}'."
        )
        observed = (
            read_observed_generation(inputs, spread["exclude_areas"])
            if spread["reference"] == "observed"
            else (None, None)
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
