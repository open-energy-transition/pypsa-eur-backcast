# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Renewable energy configuration.

See docs in https://pypsa-eur.readthedocs.io/en/latest/configuration/#renewable_cf
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scripts.lib.validation.config._base import ConfigModel


class _WindResourceConfig(ConfigModel):
    """Configuration for wind resource settings."""

    method: str = Field("wind", description="A superordinate technology type.")
    turbine: str | dict[int, str] = Field(
        ...,
        description="Specifies the turbine type and its characteristic power curve. Can be a string or a dictionary with years as keys which denote the year another turbine model becomes available.",
    )
    smooth: bool = Field(
        False,
        description="Switch to apply a gaussian kernel density smoothing to the power curve.",
    )
    add_cutout_windspeed: bool = Field(
        True,
        description="Whether to add cutout windspeed data.",
    )


class _SolarResourceConfig(BaseModel):
    """Configuration for solar resource settings."""

    method: str = Field("pv", description="A superordinate technology type.")
    panel: str | dict[int, str] = Field(
        "CSi",
        description="Specifies the solar panel technology and its characteristic attributes. Can be a string or a dictionary with years as keys which denote the year another panel model becomes available.",
    )
    orientation: dict[str, float] = Field(
        default_factory=lambda: {"slope": 35.0, "azimuth": 180.0},
        description="Panel orientation with slope and azimuth.",
    )
    tracking: str | None = Field(
        None,
        description="Tracking type (e.g., 'horizontal').",
    )


class _CorineConfig(BaseModel):
    """Configuration for CORINE land cover settings."""

    grid_codes: list[int] = Field(
        ...,
        description="Specifies areas according to CORINE Land Cover codes which are generally eligible for wind turbine placement.",
    )
    distance: float = Field(
        1000,
        description="Distance in meters to keep from areas specified in `distance_grid_codes`.",
    )
    distance_grid_codes: list[int] = Field(
        default_factory=list,
        description="Specifies areas according to CORINE Land Cover codes to which wind turbines must maintain a distance specified in the setting `distance`.",
    )


class _OnwindConfig(BaseModel):
    """Configuration for onshore wind."""

    cutout: str | list[str] = Field(
        "default", description="Specifies the weather data cutout file(s) to use."
    )
    resource: _WindResourceConfig = Field(
        default_factory=lambda: _WindResourceConfig(turbine="Vestas_V112_3MW"),
        description="Wind resource configuration.",
    )
    resource_classes: int = Field(
        1, description="Number of resource classes per clustered region."
    )
    capacity_per_sqkm: float = Field(
        3, description="Allowable density of wind turbine placement."
    )
    correction_factor: float = Field(
        1.0, description="Correction factor for capacity factor time series."
    )
    corine: bool | _CorineConfig = Field(
        default_factory=lambda: _CorineConfig(
            grid_codes=[
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                31,
                32,
            ],
            distance=1000,
            distance_grid_codes=[1, 2, 3, 4, 5, 6],
        ),
        description="CORINE land cover configuration.",
    )
    luisa: bool | dict = Field(False, description="LUISA land cover configuration.")
    natura: bool = Field(
        True,
        description="Switch to exclude `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas. Area is excluded if `true`.",
    )
    excluder_resolution: float = Field(
        100,
        description="Resolution in meters on which to perform geographical eligibility analysis.",
    )
    clip_p_max_pu: float = Field(
        0.01,
        description="To avoid too small values in the renewables` per-unit availability time series values below this threshold are set to zero.",
    )


class _OffwindConfig(BaseModel):
    """Configuration for offshore wind."""

    cutout: str | list[str] = Field(
        "default", description="Specifies the weather data cutout file(s) to use."
    )
    resource: _WindResourceConfig = Field(
        default_factory=lambda: _WindResourceConfig(
            turbine="NREL_ReferenceTurbine_2020ATB_5.5MW"
        ),
        description="Wind resource configuration.",
    )
    resource_classes: int = Field(
        1, description="Number of resource classes per clustered region."
    )
    capacity_per_sqkm: float = Field(
        2, description="Allowable density of wind turbine placement."
    )
    correction_factor: float = Field(
        0.8855, description="Correction factor for capacity factor time series."
    )
    corine: bool | list[int] = Field(
        default=False,
        description="Specifies areas according to CORINE Land Cover codes which are generally eligible for AC-connected offshore wind turbine placement.",
    )
    luisa: bool | list[int] = Field(
        False,
        description="Specifies areas according to the LUISA Base Map codes which are generally eligible for AC-connected offshore wind turbine placement.",
    )
    natura: bool = Field(
        True,
        description="Switch to exclude `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas. Area is excluded if `true`.",
    )
    ship_threshold: float = Field(
        400, description="Ship density threshold from which areas are excluded."
    )
    max_depth: float | None = Field(
        None,
        description="Maximum sea water depth in meters at which wind turbines can be built. Maritime areas with deeper waters are excluded in the process of calculating the AC-connected offshore wind potential.",
    )
    min_depth: float | None = Field(None, description="Minimum water depth in meters.")
    max_shore_distance: float | None = Field(
        None,
        description="Maximum distance to the shore in meters above which wind turbines cannot be built. Such areas are excluded in the process of calculating the AC-connected offshore wind potential.",
    )
    min_shore_distance: float | None = Field(
        None,
        description="Minimum distance to the shore in meters below which wind turbines cannot be built. Such areas close to the shore are excluded in the process of calculating the AC-connected offshore wind potential.",
    )
    excluder_resolution: float = Field(
        200,
        description="Resolution in meters on which to perform geographical eligibility analysis.",
    )
    clip_p_max_pu: float = Field(
        0.01,
        description="To avoid too small values in the renewables` per-unit availability time series values below this threshold are set to zero.",
    )
    landfall_length: float | str = Field(
        20,
        description="Fixed length of the cable connection that is onshorelandfall in km. If 'centroid', the length is calculated as the distance to centroid of the onshore bus.",
    )


class _SolarConfig(BaseModel):
    """Configuration for solar PV."""

    cutout: str | list[str] = Field(
        "default", description="Specifies the weather data cutout file(s) to use."
    )
    resource: _SolarResourceConfig = Field(
        default_factory=_SolarResourceConfig,
        description="Solar resource configuration.",
    )
    resource_classes: int = Field(
        1, description="Number of resource classes per clustered region."
    )
    capacity_per_sqkm: float = Field(
        5.1, description="Allowable density of solar panel placement."
    )
    correction_factor: float = Field(
        1.0,
        description="A correction factor for the capacity factor (availability) time series.",
    )
    corine: bool | list[int] = Field(
        default_factory=lambda: [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            26,
            31,
            32,
        ],
        description="Specifies areas according to CORINE Land Cover codes which are generally eligible for solar panel placement.",
    )
    luisa: bool | list[int] = Field(
        False,
        description="Specifies areas according to the LUISA Base Map codes which are generally eligible for solar panel placement.",
    )
    natura: bool = Field(
        True,
        description="Switch to exclude `Natura 2000 <https://en.wikipedia.org/wiki/Natura_2000>`_ natural protection areas. Area is excluded if `true`.",
    )
    excluder_resolution: float = Field(
        100,
        description="Resolution in meters on which to perform geographical eligibility analysis.",
    )
    clip_p_max_pu: float = Field(
        0.01,
        description="To avoid too small values in the renewables` per-unit availability time series values below this threshold are set to zero.",
    )


FILLING_AREAS = [
    "NO_1",
    "NO_2",
    "NO_3",
    "NO_4",
    "NO_5",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
    "FI",
    "ES",
    "PT",
    "AT",
    "IT",
    "FR",
    "BG",
    "RO",
    "HR",
]


class _HydroEntsoeBalanceConfig(BaseModel):
    """Configuration for the hydro inflow from the ENTSO-E water balance."""

    filling_first_year: dict[str, int] = Field(
        default_factory=lambda: {area: 2015 for area in FILLING_AREAS} | {"RS": 2017},
        description="ENTSO-E areas that publish the weekly filling of water reservoirs and hydro storage plants (Transparency Platform 16.1.D) with the first year whose filling is retrieved. CH publishes filling too but is in `exclude_areas`. The maximum filling from this year to the last snapshot year is the reservoir energy capacity. Defaults: first published week 2014-12-28 to 2015-02-22 for all areas except RS, which publishes from 2016-12-11 (weekly filling retrieved for 2015 to 2024 in dev/day-ahead-backcast/phase3/data/reservoir_filling_2015_2024_weekly_MWh.csv).",
    )
    exclude_areas: list[str] = Field(
        default_factory=lambda: ["CH", "XK"],
        description="ENTSO-E areas that keep the EIA based inflow because their published actual generation misses most of the hydro fleet: CH 13.1 TWh hydro and run-of-river in 2023 on the Transparency Platform against 37.0 TWh in `data/eia_hydro_annual_generation.csv`, XK 0.01 against 0.39 TWh.",
    )


class _HydroConfig(BaseModel):
    """Configuration for hydropower."""

    cutout: str | list[str] = Field(
        "default", description="Specifies the weather data cutout file(s) to use."
    )
    carriers: list[str] = Field(
        default_factory=lambda: ["ror", "PHS", "hydro"],
        description="Specifies the types of hydro power plants to build per-unit availability time series for. 'ror' stands for run-of-river plants, 'PHS' represents pumped-hydro storage, and 'hydro' stands for hydroelectric dams.",
    )
    PHS_max_hours: float = Field(
        6,
        description="Maximum state of charge capacity of the pumped-hydro storage (PHS) in terms of hours at full output capacity `p_nom`. Cf. `PyPSA documentation <https://docs.pypsa.org/latest/user-guide/components/storage-units/>`_.",
    )
    PHS_max_hours_cap: Literal["geth2015"] | None = Field(
        None,
        description="Cap the state of charge capacity of pumped-hydro storage (PHS) at the country ratio of storage energy to discharge capacity. 'geth2015' uses table 25 of Geth et al. (2015), Renew. Sust. Energy Rev. 52, 1212-1227. Countries without data keep their value.",
    )
    hydro_max_hours: str | float = Field(
        "energy_capacity_totals_by_country",
        description="Maximum state of charge capacity of the pumped-hydro storage (PHS) in terms of hours at full output capacity `p_nom` or heuristically determined. Cf. `PyPSA documentation <https://docs.pypsa.org/latest/user-guide/components/storage-units/>`_.",
    )
    flatten_dispatch: bool = Field(
        False,
        description="Consider an upper limit for the hydro dispatch. The limit is given by the average capacity factor plus the buffer given in `flatten_dispatch_buffer`.",
    )
    flatten_dispatch_buffer: float = Field(
        0.2,
        description="If `flatten_dispatch` is true, specify the value added above the average capacity factor.",
    )
    clip_min_inflow: float = Field(
        1.0,
        description="To avoid too small values in the inflow time series, values below this threshold (MW) are set to zero.",
    )
    eia_norm_year: bool | int = Field(
        False,
        description="To specify a specific year by which hydro inflow is normed that deviates from the snapshots' year.",
    )
    eia_correct_by_capacity: bool = Field(
        False,
        description="Correct EIA annual hydro generation data by installed capacity.",
    )
    eia_approximate_missing: bool = Field(
        False,
        description="Approximate hydro generation data for years not included in EIA dataset through a regression based on annual runoff.",
    )
    inflow_source: Literal["eia", "entsoe_balance"] = Field(
        "eia",
        description="Source of the run-of-river and reservoir inflow. 'eia' scales the runoff per country to the annual EIA hydro generation and splits it onto the plants by `p_nom`. 'entsoe_balance' takes the inflow per ENTSO-E bidding zone area in `backcast: area_to_bus` from the hourly actual generation of run-of-river and reservoir hydro plus the weekly change of the reservoir filling, sets the reservoir energy capacity to the maximum observed filling and the reservoir dispatch efficiency to 1. Areas without published generation keep the EIA based inflow.",
    )
    entsoe_balance: _HydroEntsoeBalanceConfig = Field(
        default_factory=_HydroEntsoeBalanceConfig,
        description="Configuration for `inflow_source: entsoe_balance`.",
    )


class RenewableConfig(BaseModel):
    """Configuration for `renewable` settings."""

    onwind: _OnwindConfig = Field(
        default_factory=_OnwindConfig,
        description="Onshore wind configuration.",
    )
    offwind_ac: _OffwindConfig = Field(
        default_factory=lambda: _OffwindConfig(
            max_depth=60,
            max_shore_distance=30000,
            landfall_length=20,
        ),
        alias="offwind-ac",
        description="Offshore wind AC configuration.",
    )
    offwind_dc: _OffwindConfig = Field(
        default_factory=lambda: _OffwindConfig(
            max_depth=60,
            min_shore_distance=30000,
            landfall_length=30,
        ),
        alias="offwind-dc",
        description="Offshore wind DC configuration.",
    )
    offwind_float: _OffwindConfig = Field(
        default_factory=lambda: _OffwindConfig(
            resource=_WindResourceConfig(turbine="NREL_ReferenceTurbine_5MW_offshore"),
            min_depth=60,
            max_depth=1000,
            landfall_length=40,
        ),
        alias="offwind-float",
        description="Floating offshore wind configuration.",
    )
    solar: _SolarConfig = Field(
        default_factory=_SolarConfig,
        description="Solar PV configuration.",
    )
    solar_hsat: _SolarConfig = Field(
        default_factory=lambda: _SolarConfig(
            resource=_SolarResourceConfig(tracking="horizontal"),
            capacity_per_sqkm=4.43,
        ),
        alias="solar-hsat",
        description="Solar PV with horizontal single-axis tracking configuration.",
    )
    hydro: _HydroConfig = Field(
        default_factory=_HydroConfig,
        description="Hydropower configuration.",
    )

    model_config = ConfigDict(populate_by_name=True)
