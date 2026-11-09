# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Derive the carrier lists of the backcast dispatch configuration."""

from typing import NamedTuple

RESERVED_KEYS = {"chp_floor"}


class DispatchCarriers(NamedTuple):
    """Carriers and settings per backcast dispatch option."""

    nuclear_monthly: bool
    nuclear_min_output: bool
    monthly: list[str]
    outages: list[str]
    budget: list[str]
    eurostat: list[str]
    fixed: list[str]
    exclude: dict[str, list[str]]
    monthly_ceiling: dict[str, bool]
    fuel_price: dict[str, float]
    min_coverage: dict[str, float]

    @property
    def uses_monthly_generation(self) -> bool:
        """Return whether the monthly actual generation is needed."""
        return self.nuclear_monthly or bool(self.monthly)

    @property
    def uses_hourly_generation(self) -> bool:
        """Return whether the hourly actual generation is needed."""
        return self.nuclear_min_output or bool(self.budget)


def dispatch_carriers(dispatch: dict) -> DispatchCarriers:
    """
    Return the carriers per option of the `backcast: dispatch` configuration.

    Nuclear with `availability: monthly` or `outages` gets the monthly
    availability, the outages replace it in areas with enough outage data.
    Carriers keep the order of `dispatch`.
    """
    carriers = {k: v for k, v in dispatch.items() if k not in RESERVED_KEYS}

    def having(key: str, *values: str) -> list[str]:
        return [c for c, d in carriers.items() if d[key] in values]

    return DispatchCarriers(
        nuclear_monthly=carriers["nuclear"]["availability"] != "none",
        nuclear_min_output=carriers["nuclear"]["min_output"] != "none",
        monthly=[c for c in having("availability", "monthly") if c != "nuclear"],
        outages=having("availability", "outages"),
        budget=having("energy", "budget", "eurostat_budget", "fixed"),
        eurostat=having("energy", "eurostat_budget", "fixed"),
        fixed=having("energy", "fixed"),
        exclude={c: d["exclude_areas"] for c, d in carriers.items()},
        monthly_ceiling={c: d["monthly_ceiling"] for c, d in carriers.items()},
        fuel_price={
            c: d["fuel_price"]
            for c, d in carriers.items()
            if d["fuel_price"] is not None
        },
        min_coverage={c: d["outage_min_coverage"] for c, d in carriers.items()},
    )
