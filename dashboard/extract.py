# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Extract the solved backcast network and its benchmark data into parquet files for the dashboard.

Usage: ``python -m dashboard.extract results/<run> [out_dir]``, the output
directory defaults to ``dashboard/cache/<run>``.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypsa
import yaml
from snakemake.utils import update_config

from scripts.lib.backcast_data import (
    align_prices,
    area_capacities,
    marginal_carriers,
    net_exchanges,
)
from scripts.lib.backcast_plotting import area_flows, area_generation
from scripts.lib.entsoe import read_area_to_bus

CACHE = Path(__file__).parent / "cache"
SIMPLIFY_TOLERANCE = 0.01
POWER_FRAMES = {"load", "generation", "flows"}


def run_config(run: str) -> dict:
    """Read the default config updated by ``config/backcast/config.backcast.yaml`` and the scenario of the run."""
    config = yaml.safe_load(Path("config/config.default.yaml").read_text())
    base = yaml.safe_load(Path("config/backcast/config.backcast.yaml").read_text())
    scenarios = yaml.safe_load(Path(base["run"]["scenarios"]["file"]).read_text())
    for override in (base, scenarios[run]):
        update_config(config, override)
    return config


def sources(run_dir: Path) -> dict[str, Path]:
    """Resolve the input paths of a run as the backcast rules do."""
    run = run_dir.name
    config = run_config(run)
    resources = Path("resources") / run
    year = pd.Timestamp(config["snapshots"]["start"]).year
    return {
        "network": run_dir / f"networks/solved_{year}.nc",
        "prices": resources / "day_ahead_prices.csv",
        "generation": resources / "entsoe_generation_hourly.csv",
        "exchanges": resources / "scheduled_exchanges.csv",
        "capacities": resources / "cross_border_capacities.csv",
        "regions": resources / "onshore_regions.geojson",
        "metrics": run_dir / "backcast/price_metrics.csv",
        "monthly_bias": run_dir / "backcast/price_monthly_bias.csv",
        "area_to_bus": Path(config["backcast"]["area_to_bus"]),
    }


def by_bus(frame: pd.DataFrame, buses: pd.Series) -> pd.DataFrame:
    """Sum the columns of a component time series by their bus."""
    return frame.T.groupby(buses[frame.columns]).sum().T


def load_frame(n: pypsa.Network) -> pd.DataFrame:
    """Return the hourly load and load shedding per bus."""
    shedding = n.generators.query("carrier == 'load'").index
    return pd.concat(
        {
            "load": by_bus(n.loads_t.p, n.loads.bus),
            "load_shedding": by_bus(n.generators_t.p[shedding], n.generators.bus),
        },
        axis=1,
        names=["kind", "bus"],
    )


def git_commit() -> str:
    """Return the short hash of the checked out commit."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame as zstd compressed parquet, storing power in MW at 0.1 MW resolution as float32."""
    if path.stem in POWER_FRAMES:
        frame = frame.round(1).astype("float32")
    frame.to_parquet(path, compression="zstd")


def main(run_dir: Path, out_dir: Path) -> None:
    """Write the dashboard extract of the run in ``run_dir`` to ``out_dir``."""
    extract(sources(run_dir), out_dir, run_dir.name)


def extract(paths: dict[str, Path], out_dir: Path, run: str) -> None:
    """Write the dashboard extract of the run with the input paths to ``out_dir``."""
    missing = [str(p) for p in paths.values() if not p.exists()]
    assert not missing, f"Missing inputs: {missing}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = pypsa.Network(paths["network"])
    area_to_bus = read_area_to_bus(paths["area_to_bus"])
    snapshots = n.snapshots
    model = n.buses_t.marginal_price[n.buses.query("carrier == 'AC'").index]
    observed = pd.read_csv(paths["prices"], index_col=0, parse_dates=True)
    model, observed = align_prices(model, observed)
    entsoe = pd.read_csv(
        paths["generation"], index_col=0, header=[0, 1], parse_dates=True
    )
    exchanges = pd.read_csv(paths["exchanges"], index_col=0, parse_dates=True)
    capacities = pd.read_csv(paths["capacities"], index_col=0, parse_dates=True)

    frames = {
        "prices": pd.concat(
            {"model": model, "observed": observed}, axis=1, names=["series", "bus"]
        ),
        "load": load_frame(n),
        "generation": pd.concat(
            {
                "model": area_generation(n, area_to_bus),
                "entsoe": entsoe.reindex(snapshots),
            },
            axis=1,
            names=["source", "area", "carrier"],
        ),
        "flows": pd.concat(
            {
                "model": area_flows(n, area_to_bus),
                "scheduled": net_exchanges(exchanges).reindex(snapshots),
                "ntc": area_capacities(capacities, area_to_bus).reindex(snapshots),
            },
            axis=1,
            names=["source", "border"],
        ),
        "marginal": marginal_carriers(n)[observed.columns].astype("category"),
        "metrics": pd.read_csv(paths["metrics"], index_col=0),
        "monthly_bias": pd.read_csv(paths["monthly_bias"], index_col=0),
    }
    for name, frame in frames.items():
        write(frame, out_dir / f"{name}.parquet")

    regions = gpd.read_file(paths["regions"])
    regions.geometry = regions.geometry.simplify(SIMPLIFY_TOLERANCE)
    regions[regions.name.isin(area_to_bus.index)].to_file(
        out_dir / "regions.geojson", driver="GeoJSON"
    )
    area_to_bus.to_csv(out_dir / "area_to_bus.csv")
    manifest = {
        "run": run,
        "year": int(snapshots[0].year),
        "commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "sources": {str(p): p.stat().st_mtime for p in paths.values()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    run_dir = Path(sys.argv[1].rstrip("/"))
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE / run_dir.name
    main(run_dir, out_dir)
