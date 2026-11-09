# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Backcast configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#backcast_cf
"""

from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, field_validator, model_validator

from scripts.lib.validation.config._base import ConfigModel
from scripts.lib.validation.config.backcast_dispatch import DispatchConfig

MAX_INTERPOLATE_PRICE_GAP_HOURS = 3

GapHours = Annotated[int, Field(ge=MAX_INTERPOLATE_PRICE_GAP_HOURS)]


class HybridCapacitiesConfig(ConfigModel):
    """Configuration for `backcast.interconnection.hybrid` settings."""

    enable: bool = Field(
        False,
        description="Raise the hourly capacity of border directions whose source is in `sources` and that connect two of the `areas` to at least `physical_factor` times `s_nom` of the lines plus `dc_physical_factor` times `p_nom` of the DC links between the two buses (the physical capacity, as with `interconnection: fixed: physical`), e.g. for the monthly maximum of scheduled exchanges in the flow-based Core region, which underestimates the available capacity. Requires `electricity: cross_border_capacities: ntc`.",
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


class ExternalConfig(ConfigModel):
    """Configuration for `backcast.areas.external` settings."""

    enable: bool = Field(
        False,
        description="Add the hourly ENTSO-E exchanges with the `areas` outside the model as fixed flows at the bus with the largest load of the bordering modelled area: net imports as a generator with `p_min_pu` equal to `p_max_pu`, net exports as a load. The borders are listed in `data/entsoe_external_borders.csv`, whose `flow` column selects scheduled exchanges (`source`) or physical flows per border with a cited reason. Hours without published exchanges count as zero and are logged as a warning per border, a border without any published value in the year raises. Areas modelled as buses raise an error, e.g. `GB`.",
    )
    areas: list[str] = Field(
        default_factory=lambda: ["UA", "MD", "TR", "RU", "RU_KGD", "BY"],
        description="ENTSO-E areas outside the model whose exchanges are fixed. `UA` returns the exchanges of the Burshtyn island (`UA_BEI`) and of `UA_DOBTPP` before the synchronisation of Ukraine in 2022. Morocco (ES-MA) is not included: the ENTSO-E Transparency Platform returns no scheduled exchanges or physical flows for `10YMA-ONE------O` or the border area `10YDOM-1010A0140` (probed 2019-06, 2021-06, 2023-06). EE-RU is not included: no scheduled exchanges are published, as Baltic trade with Russia and Belarus is scheduled on the LT borders, and its physical flows are loop flows of the synchronous BRELL ring (EE>RU 158 GWh in 2023-06 without trade).",
    )
    source: Literal["scheduled_exchange", "scheduled_exchange_total"] = Field(
        "scheduled_exchange_total",
        description="ENTSO-E scheduled commercial exchanges used: day-ahead (`scheduled_exchange`) or total including explicit allocations and intraday (`scheduled_exchange_total`). Most external borders publish only the total, e.g. BG-TR, GR-TR and SK-UA. Borders with `flow` `physical` in `data/entsoe_external_borders.csv` use the physical flows instead, e.g. FI-RU, which publishes no scheduled exchanges.",
    )


def check_borders(value: list[str]) -> None:
    """Require borders as two alphabetically sorted areas `AREA0-AREA1`."""
    invalid = [
        k
        for k in value
        if len(k.split("-")) != 2 or k.split("-") != sorted(k.split("-"))
    ]
    assert not invalid, f"Borders must be given as sorted `AREA0-AREA1`: {invalid}"


def check_ascending(value: list[float]) -> list[float]:
    """Require the lower bound not to exceed the upper bound."""
    assert value[0] <= value[1], f"Calibration factor bounds must be ascending: {value}"
    return value


FactorBounds = Annotated[list[PositiveFloat], Field(min_length=2, max_length=2)]


class AreasConfig(ConfigModel):
    """Configuration for `backcast.areas` settings."""

    area_to_bus: str = Field(
        "data/entsoe_area_to_bus.csv",
        description="CSV file with the many-to-one map from model buses (`bus`) to ENTSO-E bidding zone areas (`entsoe_area`), e.g. `data/entsoe_area_to_bus-pre2021.csv` for the Italian bidding zones before 2021, in which the Calabria bus `ITCA` maps to the virtual zone `IT_ROSN`.",
    )
    borders: str = Field(
        "data/entsoe_borders.csv",
        description="CSV file with the ENTSO-E bidding zone borders (`area0`, `area1`) whose cross-border capacities are retrieved, used with `electricity: cross_border_capacities: ntc`, e.g. `data/entsoe_borders-pre2021.csv` for the Italian bidding zones before 2021.",
    )
    external: ExternalConfig = Field(
        default_factory=ExternalConfig,
        description="Fixed hourly flows to ENTSO-E areas outside the model.",
    )


class PricesConfig(ConfigModel):
    """Configuration for `backcast.market_data.prices` settings."""

    exclude_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E bidding zone areas without published day-ahead prices for the whole snapshot range, e.g. `GB` after 2020, `BA` and `XK`, or `AL`, `ME` and `MK` whose markets started during 2023. Buses mapped to these areas in `data/entsoe_area_to_bus.csv` are omitted from the observed day-ahead prices.",
    )
    exclude_buses: list[str] = Field(
        default_factory=list,
        description="Buses omitted from the observed day-ahead prices although their ENTSO-E area publishes prices, e.g. `FR15` (Corsica), which is not part of the French day-ahead market.",
    )
    max_gap_hours: GapHours = Field(
        MAX_INTERPOLATE_PRICE_GAP_HOURS,
        description=f"Longest gap in hours of observed day-ahead prices that is tolerated. Gaps of up to {MAX_INTERPOLATE_PRICE_GAP_HOURS} hours are interpolated, longer gaps up to this limit are kept as missing values, even longer gaps raise an error.",
    )
    max_gap_hours_by_area: dict[str, GapHours] = Field(
        default_factory=dict,
        description="Longest tolerated gap in hours of observed day-ahead prices per ENTSO-E area, overriding `market_data: prices: max_gap_hours`, e.g. `IE_SEM: 25` as the ENTSO-E Transparency Platform publishes no prices for the 25 hour delivery day at the end of daylight saving time in 2022 and 2023.",
    )


class GenerationConfig(ConfigModel):
    """Configuration for `backcast.market_data.generation` settings."""

    exclude_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas without actual generation published on the ENTSO-E Transparency Platform for the snapshot years, e.g. `GB` after 2020. Their actual generation is not retrieved and their generators keep their availability with `renewables: wind: calibrate`, unless given in `renewables: wind: target_twh`, and `dispatch: <carrier>: availability: monthly`. Other areas without published data raise an error during retrieval.",
    )


class CapacitiesConfig(ConfigModel):
    """Configuration for `backcast.market_data.capacities` settings."""

    max_gap_hours: Annotated[int, Field(ge=0)] = Field(
        48,
        description="Longest gap in hours of the ENTSO-E cross-border capacity source of a border direction that is interpolated, used with `electricity: cross_border_capacities: ntc`. Longer gaps raise an error. The default covers up to two missing delivery days, e.g. of the offered capacities on the Nordic and Baltic borders in 2023.",
    )


class MarketDataConfig(ConfigModel):
    """Configuration for `backcast.market_data` settings."""

    prices: PricesConfig = Field(
        default_factory=PricesConfig,
        description="Observed day-ahead prices of the ENTSO-E Transparency Platform.",
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Actual generation of the ENTSO-E Transparency Platform.",
    )
    capacities: CapacitiesConfig = Field(
        default_factory=CapacitiesConfig,
        description="Cross-border capacities of the ENTSO-E Transparency Platform.",
    )


class ProxyConfig(ConfigModel):
    """Configuration for `backcast.interconnection.proxy` settings."""

    borders: list[str] = Field(
        default_factory=list,
        description="Bidding zone borders that use the monthly maximum of the day-ahead scheduled commercial exchanges over both directions as capacity even if ENTSO-E publishes capacities, used with `electricity: cross_border_capacities: ntc`. Given as alphabetically sorted ENTSO-E area pairs like `BE-NL`, e.g. for borders within the flow-based Core region whose published net transfer capacities are outdated. `interconnection: fixed` take precedence.",
    )
    fallback: bool = Field(
        False,
        description="Fill gaps longer than `market_data: capacities: max_gap_hours` in the ENTSO-E capacity source of a border direction with the monthly maximum of the day-ahead scheduled commercial exchanges over both directions instead of raising an error, used with `electricity: cross_border_capacities: ntc`. The source of such directions is recorded as `<source>+scheduled_exchange`. E.g. for the Nordic borders, whose offered capacities end with the start of flow-based market coupling on 2024-10-29.",
    )
    by_direction_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas whose borders with each other use the monthly maximum of the day-ahead scheduled commercial exchanges per direction as exchange proxy, instead of the maximum over both directions. As the exchanges are netted per hour, the proxy of a rarely used direction is low. E.g. for the Nordic flow-based region, where the maximum over both directions gives rarely used directions the capacity of the busy one.",
    )

    @field_validator("borders")
    @classmethod
    def check_border_keys(cls, value: list[str]) -> list[str]:
        """Require borders as two alphabetically sorted areas `AREA0-AREA1`."""
        check_borders(value)
        return value


class InterconnectionConfig(ConfigModel):
    """Configuration for `backcast.interconnection` settings, with precedence fixed over proxy over hybrid."""

    fixed: dict[str, Annotated[float, Field(ge=0)] | Literal["physical"]] = Field(
        default_factory=dict,
        description="Constant capacity in MW per direction for bidding zone borders without published ENTSO-E capacities, used with `electricity: cross_border_capacities: ntc`. Keys are alphabetically sorted ENTSO-E area pairs like `DK_1-GB`; the value `physical` keeps the physical transmission capacity of the network and `0` disconnects the areas, e.g. for interconnectors commissioned after the snapshots or connections without a market border created by the clustering.",
    )
    proxy: ProxyConfig = Field(
        default_factory=ProxyConfig,
        description="Monthly maximum of the day-ahead scheduled commercial exchanges as cross-border capacity.",
    )
    hybrid: HybridCapacitiesConfig = Field(
        default_factory=HybridCapacitiesConfig,
        description="Raise cross-border capacities from proxy sources to a share of the physical transmission capacity, used with `electricity: cross_border_capacities: ntc`.",
    )

    @field_validator("fixed")
    @classmethod
    def check_border_keys(cls, value: dict[str, float | str]) -> dict[str, float | str]:
        """Require borders as two alphabetically sorted areas `AREA0-AREA1`."""
        check_borders(list(value))
        return value


class WindConfig(ConfigModel):
    """Configuration for `backcast.renewables.wind` settings."""

    calibrate: bool = Field(
        False,
        description="Scale the onshore and offshore wind availability (`p_max_pu`) per ENTSO-E area in `data/entsoe_area_to_bus.csv` so that the available energy matches the actual generation reported by the ENTSO-E Transparency Platform plus `renewables: wind: curtailment_twh`. Only months with actual generation of the carrier are compared. One factor per area and carrier, `p_max_pu` is clipped at 1. Wind carriers with capacity but without actual generation in an area that publishes generation, e.g. offshore wind in `IE_SEM`, get zero availability. Areas without published generation are scaled to `renewables: wind: target_twh` or keep their availability, as do carriers excluded in `renewables: wind: exclude`. Requires `foresight: overnight`.",
    )
    exclude: dict[str, list[Literal["onwind", "offwind"]]] = Field(
        default_factory=dict,
        description="Wind carriers per ENTSO-E area whose availability is not calibrated with `renewables: wind: calibrate`, e.g. `NL: [onwind]`, whose reported onshore wind generation covers only part of the fleet. Wind curtailment cannot be given for these areas.",
    )
    factor_bounds: FactorBounds = Field(
        default_factory=lambda: [0.5, 2.0],
        description="Lower and upper bound of the wind calibration factors of `renewables: wind: calibrate`. A factor outside the bounds raises an error naming the area and carrier, which must then be added to `renewables: wind: exclude` or the bounds widened.",
    )
    curtailment_twh: dict[
        str,
        Annotated[float, Field(ge=0)]
        | dict[Literal["onwind", "offwind"], Annotated[float, Field(ge=0)]],
    ] = Field(
        default_factory=dict,
        description="Curtailed wind energy in TWh per ENTSO-E area added to the actual generation with `renewables: wind: calibrate`, either per carrier like `DE_LU: {onwind: 6, offwind: 4}` or as a total split between onshore and offshore wind by their actual generation. The ENTSO-E actual generation is reported after curtailment, e.g. after redispatch in `DE_LU`.",
    )
    target_twh: dict[
        str, dict[Literal["onwind", "offwind"], Annotated[float, Field(ge=0)]]
    ] = Field(
        default_factory=dict,
        description="Wind energy in TWh per carrier over all snapshots for ENTSO-E areas without published actual generation (`market_data: generation: exclude_areas`), to which `renewables: wind: calibrate` scales their availability instead of the ENTSO-E data, e.g. `GB: {onwind: 32.5, offwind: 49.6}` from DESNZ Energy Trends for 2023. Not allowed for areas with published generation or carriers in `renewables: wind: exclude`.",
    )

    @field_validator("factor_bounds")
    @classmethod
    def check_factor_bounds(cls, value: list[float]) -> list[float]:
        """Require the lower bound not to exceed the upper bound."""
        return check_ascending(value)


class SolarConfig(ConfigModel):
    """Configuration for `backcast.renewables.solar` settings."""

    calibrate: bool = Field(
        False,
        description="Scale the solar availability (`p_max_pu`) per ENTSO-E area in `data/entsoe_area_to_bus.csv` so that the available energy matches the actual solar generation reported by the ENTSO-E Transparency Platform, i.e. the grid-fed output on the same basis as the ENTSO-E load. Only months with actual solar generation are compared. One factor per area, `p_max_pu` is clipped at 1. Areas that publish generation but no solar get zero solar availability. Areas in `renewables: solar: exclude` and in `market_data: generation: exclude_areas` are not calibrated; the self-consumed share of their country in `data/solar_self_consumption.csv` is removed from their availability instead, countries without a share keep it. With `renewables: solar: unpublished`, calibrated areas add the solar generation that ENTSO-E does not publish to their target. Requires `foresight: overnight`.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas whose solar availability is not calibrated with `renewables: solar: calibrate` because ENTSO-E reports only part of their solar generation, e.g. `NL`. Their self-consumed share is removed from the availability instead.",
    )
    factor_bounds: FactorBounds = Field(
        default_factory=lambda: [0.5, 2.0],
        description="Lower and upper bound of the solar calibration factors of `renewables: solar: calibrate`. A factor outside the bounds raises an error naming the area, which must then be added to `renewables: solar: exclude` or the bounds widened.",
    )
    unpublished: bool = Field(
        False,
        description="Add the grid-fed solar generation in TWh that ENTSO-E does not publish but that serves the ENTSO-E load, per area from `data/solar_unpublished.csv` for the snapshot year, to the targets of `renewables: solar: calibrate`. The file must have rows for the snapshot year. Requires `renewables: solar: calibrate`.",
    )

    @field_validator("factor_bounds")
    @classmethod
    def check_factor_bounds(cls, value: list[float]) -> list[float]:
        """Require the lower bound not to exceed the upper bound."""
        return check_ascending(value)

    @model_validator(mode="after")
    def check_unpublished(self) -> "SolarConfig":
        """Require the calibration for the unpublished solar generation."""
        assert self.calibrate or not self.unpublished, (
            "backcast: renewables: solar: unpublished requires calibrate."
        )
        return self


class RenewablesConfig(ConfigModel):
    """Configuration for `backcast.renewables` settings."""

    wind: WindConfig = Field(
        default_factory=WindConfig,
        description="Calibration of the onshore and offshore wind availability.",
    )
    solar: SolarConfig = Field(
        default_factory=SolarConfig,
        description="Calibration of the solar availability.",
    )


class BackcastConfig(ConfigModel):
    """Configuration for `backcast` settings."""

    areas: AreasConfig = Field(
        default_factory=AreasConfig,
        description="ENTSO-E bidding zone areas, their borders and the areas outside the model.",
    )
    market_data: MarketDataConfig = Field(
        default_factory=MarketDataConfig,
        description="Observed market data of the ENTSO-E Transparency Platform.",
    )
    interconnection: InterconnectionConfig = Field(
        default_factory=InterconnectionConfig,
        description="Cross-border capacities, used with `electricity: cross_border_capacities: ntc`. A border in `fixed` takes precedence over `proxy`, which takes precedence over `hybrid`.",
    )
    renewables: RenewablesConfig = Field(
        default_factory=RenewablesConfig,
        description="Calibration of the renewable availability to the actual generation.",
    )
    dispatch: DispatchConfig = Field(
        default_factory=DispatchConfig,
        description="Availability, energy and fuel price per generator carrier, and the heat-led CHP minimum output.",
    )

    @model_validator(mode="after")
    def check_wind_target_areas(self) -> "BackcastConfig":
        """Require wind targets only for areas without published generation."""
        published = sorted(
            set(self.renewables.wind.target_twh)
            - set(self.market_data.generation.exclude_areas)
        )
        assert not published, (
            f"backcast: renewables: wind: target_twh given for areas not in market_data: generation: exclude_areas: {published}"
        )
        return self
