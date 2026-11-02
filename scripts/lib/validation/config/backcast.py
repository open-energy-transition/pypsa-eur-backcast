# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Backcast configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#backcast_cf
"""

import datetime
from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, field_validator, model_validator

from scripts.lib.validation.config._base import ConfigModel

MAX_INTERPOLATE_PRICE_GAP_HOURS = 3

GapHours = Annotated[int, Field(ge=MAX_INTERPOLATE_PRICE_GAP_HOURS)]


class HybridCapacitiesConfig(ConfigModel):
    """Configuration for `backcast.hybrid_capacities` settings."""

    enable: bool = Field(
        False,
        description="Raise the hourly capacity of border directions whose source is in `sources` and that connect two of the `areas` to at least `physical_factor` times `s_nom` of the lines plus `dc_physical_factor` times `p_nom` of the DC links between the two buses (the physical capacity, as with `fixed_border_capacities: physical`), e.g. for the monthly maximum of scheduled exchanges in the flow-based Core region, which underestimates the available capacity. Requires `electricity: cross_border_capacities: ntc`.",
    )
    physical_factor: Annotated[float, Field(gt=0)] = Field(
        0.7,
        description="Share of the nominal capacity `s_nom` of lines used as lower bound of the capacity.",
    )
    dc_physical_factor: Annotated[float, Field(gt=0)] = Field(
        0.7,
        description="Share of the nominal capacity `p_nom` of DC links used as lower bound of the capacity.",
    )
    sources: Annotated[
        list[Literal["ntc", "scheduled_exchange"]], Field(min_length=1)
    ] = Field(
        default_factory=lambda: ["scheduled_exchange"],
        description="Capacity sources of `build_cross_border_capacities` that are raised. The offered capacities of the Nordic and Baltic borders are never raised.",
    )
    areas: list[str] = Field(
        default_factory=lambda: [
            "AT",
            "BE",
            "CZ",
            "DE_LU",
            "FR",
            "HR",
            "HU",
            "NL",
            "PL",
            "RO",
            "SI",
            "SK",
        ],
        description="ENTSO-E areas of which at least one must be at an end of a raised border direction, by default the flow-based Core region.",
    )


class BackcastConfig(ConfigModel):
    """Configuration for `backcast` settings."""

    area_to_bus: str = Field(
        "data/entsoe_area_to_bus.csv",
        description="CSV file with the many-to-one map from model buses (`bus`) to ENTSO-E bidding zone areas (`entsoe_area`), e.g. `data/entsoe_area_to_bus-pre2021.csv` for the Italian bidding zones before 2021, in which the Calabria bus `ITCA` maps to the virtual zone `IT_ROSN`.",
    )
    borders: str = Field(
        "data/entsoe_borders.csv",
        description="CSV file with the ENTSO-E bidding zone borders (`area0`, `area1`) whose cross-border capacities are retrieved, used with `electricity: cross_border_capacities: ntc`, e.g. `data/entsoe_borders-pre2021.csv` for the Italian bidding zones before 2021.",
    )
    exclude_price_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E bidding zone areas without published day-ahead prices for the whole snapshot range, e.g. `GB` after 2020, `BA` and `XK`, or `AL`, `ME` and `MK` whose markets started during 2023. Buses mapped to these areas in `data/entsoe_area_to_bus.csv` are omitted from the observed day-ahead prices.",
    )
    exclude_price_buses: list[str] = Field(
        default_factory=list,
        description="Buses omitted from the observed day-ahead prices although their ENTSO-E area publishes prices, e.g. `FR15` (Corsica), which is not part of the French day-ahead market.",
    )
    max_price_gap_hours: GapHours = Field(
        MAX_INTERPOLATE_PRICE_GAP_HOURS,
        description=f"Longest gap in hours of observed day-ahead prices that is tolerated. Gaps of up to {MAX_INTERPOLATE_PRICE_GAP_HOURS} hours are interpolated, longer gaps up to this limit are kept as missing values, even longer gaps raise an error.",
    )
    max_price_gap_hours_by_area: dict[str, GapHours] = Field(
        default_factory=dict,
        description="Longest tolerated gap in hours of observed day-ahead prices per ENTSO-E area, overriding `max_price_gap_hours`, e.g. `IE_SEM: 25` as the ENTSO-E Transparency Platform publishes no prices for the 25 hour delivery day at the end of daylight saving time in 2022 and 2023.",
    )
    max_capacity_gap_hours: Annotated[int, Field(ge=0)] = Field(
        48,
        description="Longest gap in hours of the ENTSO-E cross-border capacity source of a border direction that is interpolated, used with `electricity: cross_border_capacities: ntc`. Longer gaps raise an error. The default covers up to two missing delivery days, e.g. of the offered capacities on the Nordic and Baltic borders in 2023.",
    )
    fixed_border_capacities: dict[
        str, Annotated[float, Field(ge=0)] | Literal["physical"]
    ] = Field(
        default_factory=dict,
        description="Constant capacity in MW per direction for bidding zone borders without published ENTSO-E capacities, used with `electricity: cross_border_capacities: ntc`. Keys are alphabetically sorted ENTSO-E area pairs like `DK_1-GB`; the value `physical` keeps the physical transmission capacity of the network and `0` disconnects the areas, e.g. for interconnectors commissioned after the snapshots or connections without a market border created by the clustering.",
    )

    exchange_proxy_borders: list[str] = Field(
        default_factory=list,
        description="Bidding zone borders that use the monthly maximum of the day-ahead scheduled commercial exchanges over both directions as capacity even if ENTSO-E publishes capacities, used with `electricity: cross_border_capacities: ntc`. Given as alphabetically sorted ENTSO-E area pairs like `BE-NL`, e.g. for borders within the flow-based Core region whose published net transfer capacities are outdated. `fixed_border_capacities` take precedence.",
    )
    exchange_proxy_fallback: bool = Field(
        False,
        description="Fill gaps longer than `max_capacity_gap_hours` in the ENTSO-E capacity source of a border direction with the monthly maximum of the day-ahead scheduled commercial exchanges over both directions instead of raising an error, used with `electricity: cross_border_capacities: ntc`. The source of such directions is recorded as `<source>+scheduled_exchange`. E.g. for the Nordic borders, whose offered capacities end with the start of flow-based market coupling on 2024-10-29.",
    )

    calibrate_wind: bool = Field(
        False,
        description="Scale the onshore and offshore wind availability (`p_max_pu`) per ENTSO-E area in `data/entsoe_area_to_bus.csv` so that the available energy matches the actual generation reported by the ENTSO-E Transparency Platform plus `wind_curtailment_twh`. Only months with actual generation of the carrier are compared. One factor per area and carrier, `p_max_pu` is clipped at 1. Wind carriers with capacity but without actual generation in an area that publishes generation, e.g. offshore wind in `IE_SEM`, get zero availability. Areas without published generation are scaled to `wind_target_twh` or keep their availability, as do carriers excluded in `wind_calibration_exclude`. Requires `foresight: overnight`.",
    )
    wind_calibration_exclude: dict[str, list[Literal["onwind", "offwind"]]] = Field(
        default_factory=dict,
        description="Wind carriers per ENTSO-E area whose availability is not calibrated with `calibrate_wind`, e.g. `NL: [onwind]`, whose reported onshore wind generation covers only part of the fleet. Wind curtailment cannot be given for these areas.",
    )
    wind_calibration_factor_bounds: Annotated[
        list[PositiveFloat], Field(min_length=2, max_length=2)
    ] = Field(
        default_factory=lambda: [0.5, 2.0],
        description="Lower and upper bound of the wind calibration factors of `calibrate_wind`. A factor outside the bounds raises an error naming the area and carrier, which must then be added to `wind_calibration_exclude` or the bounds widened.",
    )
    wind_curtailment_twh: dict[
        str,
        Annotated[float, Field(ge=0)]
        | dict[Literal["onwind", "offwind"], Annotated[float, Field(ge=0)]],
    ] = Field(
        default_factory=dict,
        description="Curtailed wind energy in TWh per ENTSO-E area added to the actual generation with `calibrate_wind`, either per carrier like `DE_LU: {onwind: 6, offwind: 4}` or as a total split between onshore and offshore wind by their actual generation. The ENTSO-E actual generation is reported after curtailment, e.g. after redispatch in `DE_LU`.",
    )
    wind_target_twh: dict[
        str, dict[Literal["onwind", "offwind"], Annotated[float, Field(ge=0)]]
    ] = Field(
        default_factory=dict,
        description="Wind energy in TWh per carrier over all snapshots for ENTSO-E areas without published actual generation (`exclude_generation_areas`), to which `calibrate_wind` scales their availability instead of the ENTSO-E data, e.g. `GB: {onwind: 32.5, offwind: 49.6}` from DESNZ Energy Trends for 2023. Not allowed for areas with published generation or carriers in `wind_calibration_exclude`.",
    )
    monthly_nuclear_availability: bool = Field(
        False,
        description="Set the nuclear availability (`p_max_pu`) per ENTSO-E area in `data/entsoe_area_to_bus.csv` and month to the actual monthly capacity factor, i.e. the actual generation divided by the hours with data and the nuclear capacity of the area, clipped to [0, 1]. After the last operating date in `nuclear_phase_out` the availability is zero. Every month before it must have actual generation, otherwise an error is raised. Areas in `exclude_generation_areas` keep their availability. Requires `foresight: overnight`.",
    )
    monthly_availability_carriers: list[str] = Field(
        default_factory=list,
        description="Further generator carriers whose availability is set per ENTSO-E area and month to the actual capacity factor like `monthly_nuclear_availability`, e.g. `[lignite, biomass, waste]` when these run at full availability in the model but not in reality. Generators of areas without published generation of the carrier, e.g. biomass in `SE_3`, and areas in `exclude_generation_areas` keep their availability. Requires `foresight: overnight`.",
    )
    monthly_availability_exclude: dict[str, list[str]] = Field(
        default_factory=dict,
        description="ENTSO-E areas per carrier of `monthly_availability_carriers` kept at their availability instead of the monthly capacity factor, e.g. `{coal: [PL]}` for a fleet used flexibly as a swing exporter, whose monthly mean capacity factor would wrongly flatten its hourly dispatch and distort the prices of its trading partners.",
    )
    nuclear_area_map: dict[str, str] = Field(
        default_factory=dict,
        description="ENTSO-E areas whose nuclear generators are calibrated with `monthly_nuclear_availability` from the actual generation of another area, pooling their capacity, e.g. `SE_4: SE_3` when nuclear generators located in the clustered `SE04` region report their generation to `SE_3`.",
    )
    nuclear_phase_out: dict[str, str] = Field(
        default_factory=dict,
        description='Last operating date of the nuclear generators per ENTSO-E area with `monthly_nuclear_availability`, after which their availability is zero, given as quoted ISO date, e.g. `DE_LU: "2023-04-15"` for the German nuclear exit.',
    )
    exclude_generation_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas without actual generation published on the ENTSO-E Transparency Platform for the snapshot years, e.g. `GB` after 2020. Their actual generation is not retrieved and their generators keep their availability with `calibrate_wind`, unless given in `wind_target_twh`, and `monthly_nuclear_availability`. Other areas without published data raise an error during retrieval.",
    )
    hybrid_capacities: HybridCapacitiesConfig = Field(
        default_factory=HybridCapacitiesConfig,
        description="Raise cross-border capacities from proxy sources to a share of the physical transmission capacity, used with `electricity: cross_border_capacities: ntc`.",
    )

    @field_validator("fixed_border_capacities", "exchange_proxy_borders")
    @classmethod
    def check_border_keys(
        cls, value: dict[str, float | str] | list[str]
    ) -> dict[str, float | str] | list[str]:
        """Require borders as two alphabetically sorted areas `AREA0-AREA1`."""
        invalid = [
            k
            for k in value
            if len(k.split("-")) != 2 or k.split("-") != sorted(k.split("-"))
        ]
        assert not invalid, f"Borders must be given as sorted `AREA0-AREA1`: {invalid}"
        return value

    @field_validator("nuclear_phase_out")
    @classmethod
    def check_phase_out_dates(cls, value: dict[str, str]) -> dict[str, str]:
        """Require ISO dates like `2023-04-15`."""
        for date in value.values():
            datetime.date.fromisoformat(date)
        return value

    @field_validator("wind_calibration_factor_bounds")
    @classmethod
    def check_factor_bounds(cls, value: list[float]) -> list[float]:
        """Require the lower bound not to exceed the upper bound."""
        assert value[0] <= value[1], (
            f"Wind calibration factor bounds must be ascending: {value}"
        )
        return value

    @model_validator(mode="after")
    def check_wind_target_areas(self) -> "BackcastConfig":
        """Require wind targets only for areas without published generation."""
        published = sorted(
            set(self.wind_target_twh) - set(self.exclude_generation_areas)
        )
        assert not published, (
            f"backcast: wind_target_twh given for areas not in exclude_generation_areas: {published}"
        )
        return self
