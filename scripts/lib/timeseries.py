# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Shared helpers for gaps in observed time series.
"""

import pandas as pd


def gap_lengths(series: pd.Series) -> pd.Series:
    """Return the length of the gap each missing value belongs to, 0 where the value is present."""
    na = series.isna()
    return na.groupby((~na).cumsum()).transform("sum").where(na, 0)


def fill_short_gaps(
    data: pd.DataFrame,
    max_interpolate: int,
    max_gap: int,
    max_gap_by_column: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Interpolate gaps up to `max_interpolate` steps, keep gaps up to `max_gap` (or the column specific `max_gap_by_column`) as NaN and raise on longer gaps."""
    lengths = data.apply(gap_lengths)
    limits = pd.Series(max_gap_by_column or {}, dtype=int).reindex(
        data.columns, fill_value=max_gap
    )
    long_gaps = lengths.max()[lambda s: s > limits]
    assert long_gaps.empty, f"Gaps longer than {max_gap} hours: {long_gaps.to_dict()}"
    return data.interpolate(limit_direction="both").where(lengths <= max_interpolate)
