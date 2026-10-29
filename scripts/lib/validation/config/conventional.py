# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Conventional generators configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#conventional_cf
"""

import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.lib.validation.config._base import ConfigModel


class _EfficiencyParameters(BaseModel):
    """Parameters of the linear efficiency heuristic for one carrier."""

    efficiency: float = Field(description="Efficiency of a plant built in `year`.")
    slope: float = Field(description="Efficiency gain per later build year.")
    year: int = Field(description="Reference build year of `efficiency`.")
    max: float = Field(description="Upper bound of the estimated efficiency.")


class _EstimateEfficienciesConfig(BaseModel):
    """Configuration for `conventional.estimate_efficiencies` settings."""

    enable: bool = Field(
        False,
        description="Estimate missing plant-level efficiencies from a carrier- and age-dependent linear heuristic. The build year is the year of the last retrofit or, if unknown, of commissioning. Plants without build year or carriers without parameters fall back to the technology cost data.",
    )
    reference_year: int = Field(
        2025,
        description="Year in which the age of power plants is evaluated for degradation.",
    )
    degradation_start: int = Field(
        10,
        description="Age in years after which efficiency degradation starts.",
    )
    degradation_rate: float = Field(
        0.001,
        description="Relative efficiency loss per year of age after `degradation_start`.",
    )
    parameters: dict[str, _EfficiencyParameters] = Field(
        default_factory=lambda: {
            carrier: _EfficiencyParameters(efficiency=e, slope=s, year=y, max=m)
            for carrier, (e, s, y, m) in {
                "lignite": (0.25, 0.003, 1960, 0.42),
                "coal": (0.28, 0.003, 1960, 0.44),
                "CCGT": (0.40, 0.004, 1980, 0.60),
                "OCGT": (0.28, 0.003, 1970, 0.41),
                "oil": (0.28, 0.002, 1960, 0.38),
                "nuclear": (0.33, 0.0, 1960, 0.33),
            }.items()
        },
        description="Per carrier: efficiency `efficiency` of a plant built in `year`, rising by `slope` per later build year up to `max`.",
    )


class _MarginalCostSpreadConfig(BaseModel):
    """Configuration for `conventional.marginal_cost_spread` settings."""

    enable: bool = Field(
        False,
        description="Let the marginal cost of each generator of `carriers` rise linearly with its output, from `mc * (1 - 2 * utilisation * spread)` at zero output to `mc * (1 + 2 * (1 - utilisation) * spread)` at `p_nom`, where `mc` is the mean marginal cost of the generator over all snapshots. This approximates the spread of heat rates and bids of the units behind an aggregated generator. It sets `marginal_cost_quadratic` and requires a solver supporting quadratic objectives. Applied after all fuel and emission costs are added.",
    )
    carriers: list[str] = Field(
        default_factory=lambda: ["CCGT", "OCGT", "coal", "lignite", "oil"],
        description="Generator carriers whose marginal cost is spread.",
    )
    spread: float = Field(
        0.3,
        ge=0,
        description="Half the rise of the marginal cost from zero output to `p_nom` relative to the mean marginal cost.",
    )
    utilisation: float = Field(
        0.35,
        ge=0,
        le=1,
        description="Output relative to `p_nom` at which the marginal cost equals its mean.",
    )
    reference: Literal["nameplate", "observed"] = Field(
        "nameplate",
        description="Reference output of the spread. `nameplate` spreads the marginal cost over `p_nom` with `utilisation`. `observed` spreads it over `p_nom` scaled by the observed ENTSO-E peak generation relative to the summed `p_nom` per bidding zone area and carrier group (gas, coal, lignite, oil), capped at 1, with the observed mean relative to the observed peak as utilisation. Generators without observed generation keep `p_nom` and `utilisation`. Requires `backcast.area_to_bus` and the ENTSO-E generation data.",
    )
    exclude_areas: list[str] = Field(
        default_factory=list,
        description="ENTSO-E areas whose observed generation is not used for the `observed` reference, e.g. `[IE_SEM]`.",
    )


class _GasPriceCapConfig(BaseModel):
    """Configuration for `conventional.gas_price_cap` settings."""

    enable: bool = Field(
        False,
        description="Cap the gas price of the CCGT and OCGT generators in `countries`, like the Iberian exception of 2022-2023, which capped the gas price for power generation in Spain and Portugal. The fuel price is clipped before the marginal cost and its spread are computed. Requires `dynamic_fuel_price`.",
    )
    countries: list[str] = Field(
        default_factory=list,
        description="Countries whose gas-fired generators get the capped gas price, e.g. `[ES, PT]`.",
    )
    cap: dict[str, Annotated[float, Field(ge=0)]] = Field(
        default_factory=dict,
        description='Gas price cap in EUR/MWh_th from a quoted ISO date until the next one, e.g. `{"2023-01-01": 45, "2023-01-14": 50}`. The gas price is uncapped before the first date.',
    )

    @field_validator("cap")
    @classmethod
    def check_cap_dates(cls, value: dict[str, float]) -> dict[str, float]:
        """Require ISO dates like `2023-01-14`."""
        for date in value:
            datetime.date.fromisoformat(date)
        return value

    @model_validator(mode="after")
    def check_complete(self) -> "_GasPriceCapConfig":
        """Require countries and caps when enabled."""
        assert not self.enable or (self.countries and self.cap), (
            "conventional: gas_price_cap requires countries and cap when enabled."
        )
        return self


class _DailyGasPriceConfig(BaseModel):
    """Configuration for `conventional.daily_gas_price` settings."""

    enable: bool = Field(
        False,
        description="Replace the monthly World Bank gas price of the CCGT and OCGT generators by the daily price of their gas hub. Each gas day applies from 06:00 CET. Oil and coal keep the monthly prices. Requires `dynamic_fuel_price` and `fuel_price_nominal`.",
    )
    default_hub: Literal["THE", "MIBGAS"] | None = Field(
        "THE",
        description="Hub for countries not in `hubs`. `THE` is the daily average price of exchange trades at Trading Hub Europe (DE), available from gas day 2021-10-01. `MIBGAS` is the MIBGAS-ES day-ahead index (ES/PT). With `null`, countries not in `hubs` keep the monthly price.",
    )
    hubs: dict[str, Literal["THE", "MIBGAS"]] = Field(
        default_factory=dict,
        description="Gas hub per country, e.g. `{ES: MIBGAS, PT: MIBGAS}`.",
    )


class ConventionalConfig(ConfigModel):
    """Configuration for `conventional` settings."""

    model_config = ConfigDict(extra="allow")

    estimate_efficiencies: _EstimateEfficienciesConfig = Field(
        default_factory=_EstimateEfficienciesConfig,
        description="Estimation of missing plant-level efficiencies.",
    )
    marginal_cost_spread: _MarginalCostSpreadConfig = Field(
        default_factory=_MarginalCostSpreadConfig,
        description="Linear increase of the marginal cost of thermal generators with their output.",
    )
    country_fuel_price: dict[
        str, dict[str, dict[str, Annotated[float, Field(ge=0)]]]
    ] = Field(
        default_factory=dict,
        description='Fuel price in EUR/MWh_th per carrier and country from a quoted ISO date until the next one, replacing the monthly price, e.g. `{coal: {PL: {"2022-01-01": 15.25}}}` for domestic coal sold to Polish power plants below the international price. The price is unchanged before the first date. Applied before the marginal cost and its spread are computed. Requires `dynamic_fuel_price` and `fuel_price_nominal`.',
    )
    gas_price_cap: _GasPriceCapConfig = Field(
        default_factory=_GasPriceCapConfig,
        description="Cap of the gas price of gas-fired generators per country and period.",
    )
    daily_gas_price: _DailyGasPriceConfig = Field(
        default_factory=_DailyGasPriceConfig,
        description="Daily gas prices per hub instead of the monthly World Bank gas price.",
    )
    unit_commitment: bool = Field(
        False,
        description="Allow the overwrite of ramp_limit_up, ramp_limit_start_up, ramp_limit_shut_down, p_min_pu, min_up_time, min_down_time, and start_up_cost of conventional generators. Refer to the CSV file 'unit_commitment.csv'.",
    )
    dynamic_fuel_price: bool = Field(
        False,
        description="Consider the monthly fluctuating fuel prices for each conventional generator. Refer to the CSV file 'data/validation/monthly_fuel_price.csv'.",
    )
    fuel_price_rolling_window: int = Field(
        6,
        description="Monthly rolling mean window for fossil fuel prices smoothing.",
        ge=1,
    )
    fuel_price_nominal: bool = Field(
        False,
        description="Convert fossil fuel prices to nominal EUR using yearly exchange rates instead of deflating them to EUR2020.",
    )
    nuclear: dict[str, str | float] = Field(
        default_factory=lambda: {"p_max_pu": "data/nuclear_p_max_pu.csv"},
        description="For any carrier/technology overwrite attributes as listed below.",
    )

    @model_validator(mode="after")
    def check_gas_price_cap_dynamic(self) -> "ConventionalConfig":
        """Require dynamic fuel prices for the gas price cap."""
        assert self.dynamic_fuel_price or not self.gas_price_cap.enable, (
            "conventional: gas_price_cap requires dynamic_fuel_price."
        )
        return self

    @field_validator("country_fuel_price")
    @classmethod
    def check_country_fuel_price_dates(
        cls, value: dict[str, dict[str, dict[str, float]]]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Require ISO dates like `2022-01-01`."""
        for prices in value.values():
            for steps in prices.values():
                for date in steps:
                    datetime.date.fromisoformat(date)
        return value

    @model_validator(mode="after")
    def check_country_fuel_price_nominal(self) -> "ConventionalConfig":
        """Require nominal monthly fuel prices for the nominal country fuel prices."""
        assert not self.country_fuel_price or (
            self.dynamic_fuel_price and self.fuel_price_nominal
        ), (
            "conventional: country_fuel_price requires dynamic_fuel_price and fuel_price_nominal."
        )
        return self

    @model_validator(mode="after")
    def check_daily_gas_price_nominal(self) -> "ConventionalConfig":
        """Require nominal monthly fuel prices for the nominal daily gas prices."""
        assert not self.daily_gas_price.enable or (
            self.dynamic_fuel_price and self.fuel_price_nominal
        ), (
            "conventional: daily_gas_price requires dynamic_fuel_price and fuel_price_nominal."
        )
        return self
