# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Backcast dispatch configuration per generator carrier.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#backcast_cf
"""

import datetime
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from scripts.lib.validation.config._base import ConfigModel

EUROSTAT_CARRIERS = ["biomass"]


class CarrierDispatch(ConfigModel):
    """Configuration for `backcast.dispatch.<carrier>` settings."""

    availability: Literal["none", "monthly", "outages"] = Field(
        "none",
        description="Availability (`p_max_pu`) of the generators of the carrier. `monthly` sets it per ENTSO-E area in `areas: area_to_bus` and month to the actual capacity factor, i.e. the actual generation divided by the hours with data and the capacity of the carrier in the area, clipped to [0, 1], e.g. for lignite, biomass and waste when these run at full availability in the model but not in reality. Generators of areas without published generation of the carrier, e.g. biomass in `SE_3`, of areas in `exclude_areas` and of areas in `market_data: generation: exclude_areas` keep their availability. `outages` sets it per bus to one minus the hourly unavailable share of their capacity from the unavailability of generation and production units published on the ENTSO-E Transparency Platform (planned and forced outages). The production units are matched to powerplantmatching by EIC code, nuclear units without one by name and capacity. Areas below `outage_min_coverage` keep their availability. Requires `foresight: overnight`.",
    )
    energy: Literal["none", "budget", "eurostat_budget", "fixed"] = Field(
        "none",
        description="Energy of the generators of the carrier over the snapshots. `budget` caps it (`e_sum_max`) per ENTSO-E area at the actual generation of the area, scaled per month from the hours with data to all hours (every month needs data), shared by `p_nom` and raised to the must-run energy of `p_min_pu`. `eurostat_budget` raises the energy per area to the larger of the actual generation and the Eurostat `nrg_bal_peh` gross electricity production of all producers including autoproducers in the snapshot year (biomass: bioenergy `BIOE` minus renewable municipal waste `W6210`), shared by `p_nom` over the generators of the carrier in each country; ENTSO-E misses most autoproducer output, e.g. FI pulp mills. `fixed` fixes the output per area (`p_min_pu` = `p_max_pu`) to the hourly actual generation scaled by the energy of `eurostat_budget` over the actual energy, shared by `p_nom` and capped at the capacity of the area, instead of capping it by `e_sum_max`, for price-taking heat-led fleets such as biomass CHP whose binding annual cap would set the price; hours without actual generation keep their limits. `eurostat_budget` and `fixed` are available for biomass only. Areas in `exclude_areas` and areas without actual generation of the carrier keep their limits. Not allowed with `availability: monthly`. Requires `foresight: overnight`.",
    )
    monthly_ceiling: bool = Field(
        True,
        description="Clip the availability (`p_max_pu`) of the carrier with `energy: budget` or `eurostat_budget` per ENTSO-E area to the peak hourly actual generation of the month over the capacity of the area. With `energy: fixed` the output is not clipped.",
    )
    exclude_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas kept at their availability instead of the monthly capacity factor of `availability: monthly`, or at their limits instead of the energy of `energy`, e.g. `[PL]` for a coal fleet used flexibly as a swing exporter, whose monthly mean capacity factor would wrongly flatten its hourly dispatch and distort the prices of its trading partners.",
    )
    fuel_price: Annotated[float, Field(ge=0)] | None = Field(
        None,
        description="Fuel cost in EUR/MWh_th replacing the cost data of the technology named like the carrier, e.g. `30.68` for biomass. `null` keeps the cost data. With `conventional: dynamic_fuel_price`, it applies to the carriers without monthly fuel prices.",
    )
    outage_min_coverage: Annotated[float, Field(ge=0, le=1)] = Field(
        0.0,
        description="Minimum share of the capacity of the carrier in an ENTSO-E area matched to production units with outage data with `availability: outages`, below which the generators keep their availability.",
    )

    @model_validator(mode="after")
    def check_energy_availability(self) -> "CarrierDispatch":
        """Require no energy budget with the monthly availability."""
        assert self.energy == "none" or self.availability != "monthly", (
            "backcast: dispatch: energy requires availability other than monthly."
        )
        return self


class NuclearDispatch(CarrierDispatch):
    """Configuration for `backcast.dispatch.nuclear` settings."""

    availability: Literal["none", "monthly", "outages"] = Field(
        "none",
        description="Availability (`p_max_pu`) of the nuclear generators. `monthly` sets it per ENTSO-E area in `areas: area_to_bus` and month to the actual monthly capacity factor, i.e. the actual generation divided by the hours with data and the nuclear capacity of the area, clipped to [0, 1]. After the last operating date in `phase_out` the availability is zero. Every month before it must have actual generation, otherwise an error is raised. Areas in `exclude_areas` and in `market_data: generation: exclude_areas` keep their availability. `outages` sets the monthly availability and replaces it per bus with one minus the hourly unavailable share of the capacity from the ENTSO-E unavailability of production units in areas with enough nuclear outage data (`outage_min_coverage`). Outages miss units closed during the year, capacity without reported outages and modulation, so the availability from outages is overstated. Requires `foresight: overnight`.",
    )
    min_output: Literal["none", "monthly_min"] = Field(
        "none",
        description="Nuclear minimum output (`p_min_pu`). `monthly_min` sets it per ENTSO-E area and month to the lowest hourly actual generation of the month over the nuclear capacity of the area, clipped to [0, 1], and zero in months without actual generation. Areas without actual nuclear generation keep `conventional: nuclear: p_min_pu`, areas in `load_following` get their share of the availability instead. Generators of the keys of `area_map` are pooled with the area they map to. Requires `foresight: overnight`.",
    )
    load_following: dict[str, Annotated[float, Field(gt=0, le=1)]] = Field(
        default_factory=dict,
        description="Minimum output of the nuclear generators per ENTSO-E area as share of their availability (`p_max_pu`) with `min_output: monthly_min`, for fleets that follow the load instead of running at their observed minimum, e.g. `FR: 0.2`.",
    )
    phase_out: dict[str, str] = Field(
        default_factory=dict,
        description='Last operating date of the nuclear generators per ENTSO-E area with `availability: monthly` or `outages`, after which their availability is zero, given as quoted ISO date, e.g. `DE_LU: "2023-04-15"` for the German nuclear exit.',
    )
    outage_min_coverage: Annotated[float, Field(ge=0, le=1)] = Field(
        0.9,
        description="Minimum share of the capacity of the carrier in an ENTSO-E area matched to production units with outage data with `availability: outages`, below which the generators keep their availability.",
    )
    area_map: dict[str, str] = Field(
        default_factory=dict,
        description="ENTSO-E areas whose nuclear generators are calibrated with `availability: monthly` and `min_output: monthly_min` from the actual generation of another area, pooling their capacity, e.g. `SE_4: SE_3` when nuclear generators located in the clustered `SE04` region report their generation to `SE_3`.",
    )

    @field_validator("phase_out")
    @classmethod
    def check_phase_out_dates(cls, value: dict[str, str]) -> dict[str, str]:
        """Require ISO dates like `2023-04-15`."""
        for date in value.values():
            datetime.date.fromisoformat(date)
        return value


ChpFuel = Literal["gas", "coal", "lignite", "oil", "biomass", "waste"]


class ChpFloorConfig(ConfigModel):
    """Configuration for `backcast.dispatch.chp_floor` settings."""

    enable: bool = Field(
        False,
        description="Add an hourly heat-led minimum output (`p_min_pu`) to the conventional generators of each country and fuel group. The annual floor energy is the minimum of the back-pressure ratio `c_b` of the CHP technology in the cost data times the main-activity CHP heat output and the main-activity CHP electricity output in Eurostat `nrg_bal_peh` for the snapshot year or `year`. It follows a daily heat demand profile from the cutout temperature, weighted by the urban population.",
    )
    year: int | None = Field(
        None,
        description="Year of the Eurostat `nrg_bal_peh` CHP output that sets the annual floor energy. Defaults to the snapshot year. Set it for snapshot years Eurostat does not cover yet; the heat profile still follows the snapshot weather.",
    )
    heat_threshold: float = Field(
        15.0,
        description="Daily mean temperature in degree Celsius below which space heating demand arises (heating degree days).",
    )
    hot_water_share: Annotated[float, Field(ge=0, lt=1)] = Field(
        0.2,
        description="Share of the annual heat demand that is flat over the year (hot water and network losses). Eurostat `nrg_d_hhq` gives a water heating share of household derived heat of 0.23 (2019) and 0.25 (2023) for the EU27, with a country median of 0.17.",
    )
    fuels: list[ChpFuel] = Field(
        default_factory=lambda: ["gas", "coal", "lignite", "oil", "biomass", "waste"],
        description="Fuel groups that receive the floor: gas (CCGT, OCGT), coal, lignite, oil, biomass and waste. Oil counts marketed oil products but not refinery gas and other oil products, which fire refinery-sited plants raising process steam that does not follow heating degree days.",
    )
    replace_p_min_pu: list[ChpFuel] = Field(
        default_factory=lambda: ["biomass", "waste"],
        description="Fuel groups whose flat `conventional: <carrier>: p_min_pu` is replaced by the floor in countries with a floor. Countries without Eurostat data keep the flat value.",
    )


def carrier_field(carrier: str) -> Any:
    """Return the dispatch field of a non-nuclear `carrier`."""
    return Field(
        default_factory=CarrierDispatch,
        description=f"Dispatch of the {carrier} generators.",
    )


class DispatchConfig(ConfigModel):
    """Configuration for `backcast.dispatch` settings, keyed by the generator carriers of the network."""

    model_config = ConfigDict(extra="forbid")

    nuclear: NuclearDispatch = Field(
        default_factory=NuclearDispatch,
        description="Dispatch of the nuclear generators.",
    )
    CCGT: CarrierDispatch = carrier_field("CCGT")
    OCGT: CarrierDispatch = carrier_field("OCGT")
    coal: CarrierDispatch = carrier_field("coal")
    lignite: CarrierDispatch = carrier_field("lignite")
    oil: CarrierDispatch = carrier_field("oil")
    biomass: CarrierDispatch = carrier_field("biomass")
    waste: CarrierDispatch = carrier_field("waste")
    chp_floor: ChpFloorConfig = Field(
        default_factory=ChpFloorConfig,
        description="Heat-led minimum output of combined heat and power plants. A reserved key, not a carrier.",
    )

    @model_validator(mode="after")
    def check_eurostat_carriers(self) -> "DispatchConfig":
        """Require the Eurostat energy only for carriers with Eurostat data."""
        invalid = sorted(
            carrier
            for carrier, dispatch in self.items()
            if isinstance(dispatch, CarrierDispatch)
            and dispatch.energy in ("eurostat_budget", "fixed")
            and carrier not in EUROSTAT_CARRIERS
        )
        assert not invalid, (
            f"backcast: dispatch: energy eurostat_budget and fixed are available for {EUROSTAT_CARRIERS} only, not {invalid}."
        )
        return self
