<!--
SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
SPDX-License-Identifier: CC-BY-4.0
-->

# PyPSA-Eur Backcast: day-ahead electricity prices for Europe

> [!WARNING]
> Work in progress. This fork of [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) is under active development.
> Configs, results and interfaces may change without notice. The original PyPSA-Eur README follows below.

## Target

This project backcasts hourly day-ahead electricity prices for Europe with PyPSA-Eur.
The model has one bus per bidding zone, and zones trade over NTC links (net transfer capacities published by ENTSO-E).
All capacities are fixed to the historical fleet, so the model only optimises dispatch.
The model price is the marginal price of each bus, scored against ENTSO-E day-ahead prices.
2023 is the main year and 2019 the validation year.
A refinement is kept only if it lowers the median MAE (mean absolute error) in both years.

Median over scored model buses, EUR/MWh (43 buses in 2023, 40 in 2019):

| run | 2023 MAE / r / abs bias | 2019 MAE / r / abs bias |
|---|---|---|
| baseline (2019: plain config) | 28.5 / 0.61 / 12.3 | 9.93 / 0.567 / 4.17 |
| current | 21.03 / 0.752 / 6.72 | 7.88 / 0.707 / 4.88 |

The changes against PyPSA-Eur are listed in the release notes (`doc/release_notes.md`).

## How to run

```bash
# ENTSOE_API_TOKEN must be set (retrieval rules fail fast without it)
pixi run backcast 2023                                         # snakemake metrics and figures, -j8 --resources entsoe=2
pixi run backcast 2019
pixi run backcast 2023 "-n --rerun-triggers mtime"             # extra snakemake arguments as one quoted string
pixi run -e dashboard backcast-extract results/backcast-2023   # writes dashboard/cache/backcast-2023
pixi run -e dashboard backcast-dashboard                       # streamlit run dashboard/app.py
```

- After code changes, `--rerun-triggers mtime --forcerun compose_network` avoids re-running the ENTSO-E downloads.
- The model shared by all years is `config/backcast/config.backcast.yaml`. Each year is a scenario `backcast-<year>` in `config/backcast/scenarios.yaml` with only the settings that differ by year.
- `pixi run backcast 2019-plain` runs the unrefined 2019 reference.
- Solver: Gurobi. The marginal cost spread makes the problem a QP (optimisation with a squared cost term).
- Outputs: `results/backcast-<year>/backcast/{price_metrics.csv, price_monthly_bias.csv, price_evaluation.pdf, figures/}`.

## Adding a backcast year

Weather cutouts (`europe-<year>-sarah3-era5`) exist for 2019 to 2025, so these are the years that can be backcast. The IRENA capacities in powerplantmatching and the EIA hydro statistics in `data/eia_hydro_annual_generation.csv` end in 2024.
The years are scenarios of one model (PyPSA-Eur [scenario management](https://pypsa-eur.readthedocs.io/en/latest/configuration/#run_cf)).
The model shared by all years is `config/backcast/config.backcast.yaml`, so a model change applies to every year.
Each year is an entry `backcast-<year>` in `config/backcast/scenarios.yaml` that holds only what differs by year.
`pixi run backcast <year>` selects the scenario and sets `planning_horizons` to the year, which a scenario cannot change.

1. Copy the entry of the closest year and replace the year in the snapshots, cutout, `powerplants_filter`, `custom_powerplants`, `estimate_renewable_capacities: year` and `estimate_efficiencies: reference_year`.
2. Set the bidding zones: before 2021 `backcast: area_to_bus: data/entsoe_area_to_bus-pre2021.csv` and `borders: data/entsoe_borders-pre2021.csv` (Rossano instead of Calabria), from 2021 the defaults.
3. Set the year facts and cite each source in a comment:
   `exclude_price_areas` (areas without EUR prices or markets), `exclude_generation_areas`,
   `fixed_border_capacities` (interconnectors commissioned during or after the year),
   `wind_target_twh` (GB from DUKES minus Northern Ireland wind), `wind_curtailment_twh` (DE_LU from the BNetzA Monitoringbericht),
   `nuclear_phase_out`, `conventional: gas_price_cap` (ES and PT from 2022-06-15 to 2023-12-31) and `conventional: daily_gas_price` (THE from 2021-10, MIBGAS for ES and PT).
   From 2024 set `load: supplement_synthetic: false`, as the synthetic load ends in 2023.
4. Run `pytest test/test_backcast_configs.py`, which checks every scenario against the schema, its year and the area map, then `pixi run backcast <year>`.
   Missing ENTSO-E data stops the build with the area or border to add to the exclude lists, `fixed_border_capacities` or `wind_calibration_exclude`.
   Borders whose capacity source ends within the year can use `backcast: exchange_proxy_fallback`. Wind calibration factors just outside `wind_calibration_factor_bounds` that match the other years (e.g. AT onwind about 2, DK_2 onwind about 0.4) can widen the bounds of that year.

## Dashboard

The results are published as an interactive dashboard at <https://pypsa-eur-backcast.streamlit.app/>.
It compares modelled prices, dispatch and cross-border exchanges with ENTSO-E data per bidding zone.
The pages are The story, Explore a country, Trade between countries and How it works.

The app (`dashboard/app.py`) runs on Streamlit Community Cloud from the branch `master`.
Code changes pushed to that branch are deployed automatically.
The app installs only `dashboard/requirements.txt`, so code under `dashboard/` must not import the full PyPSA-Eur stack (pypsa, atlite, snakemake, `scripts/_helpers.py`).

The data is not in git. On first start the app downloads `dashboard-cache.tar.gz` from the release
[`dashboard-data`](https://github.com/open-energy-transition/pypsa-eur-backcast/releases/tag/dashboard-data) into `dashboard/cache/`.
To update the data after new runs:

```bash
pixi run -e dashboard backcast-extract results/backcast-2023   # repeat for every run to publish
pixi run -e dashboard backcast-pack                            # writes dashboard-cache.tar.gz from dashboard/cache
gh release upload dashboard-data dashboard-cache.tar.gz --clobber -R open-energy-transition/pypsa-eur-backcast
```

Then reboot the app in the Streamlit Community Cloud settings: a running app keeps its downloaded data.
Every run in `dashboard/cache/` is published, so remove local runs you do not want to share before packing.

## Known limitations and open items

- Price spikes are too low. Dispatch in the observed top 5 % hours matches; the gap is a bid markup that a cost-based model cannot produce.
- No hours at or below 0 EUR/MWh. Nuclear and biomass throttle bands and Nordic reservoir hydro absorb every surplus.
  Observed share 3.0 % of hours (2023), model 0.3 %.
- Germany exports too much: model net export +21.5 TWh vs scheduled -14.5 TWh (2023), +49.5 vs +31.9 TWh (2019).
- Finland is too cheap: bias -18.5 EUR/MWh in 2023.
- Level errors: in 2023 most continental zones are too expensive (Central West Europe +8 to +10, Baltics +11 to +12, Iberia +19 EUR/MWh),
  while Finland (-18.5), Ireland (-11.5) and Sicily (-9.4) are too cheap. In 2019 Sicily (-17.7) and Greece (-11.7) are too cheap.
- The Norwegian zones NO_1, NO_2 and NO_5 separate in 44 % of 2023 model hours (observed 38 %). The hydro inflow split per zone is uncertain.
- The south-east chain (HU, SK, HR) shares one price more often than the market. The Baltics and other edge zones miss the daily shape.
- 2021 prices of BG and RO are published in local currency and are not scored yet.
- The ENTSO-E NTC download is slow (98 border retrievals per year). `backcast.max_capacity_gap_hours: 48` equals the longest 2023 gap.
- Only one Gurobi licence is used, so solves run one after another.
- Biomass and solar calibration should be re-tested once the level errors (scarcity, hydro water values) are fixed.
- Fixes meant for upstream PyPSA-Eur (CH load dates, TES ratio constraints, bidding-zone shapes) and a powerplantmatching data issue for Dutch CCGTs are not filed yet.

---

[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pypsa/pypsa-eur?include_prereleases)](https://github.com/PyPSA/pypsa-eur/releases)
[![Documentation](https://readthedocs.org/projects/pypsa-eur/badge/?version=latest)](https://pypsa-eur.readthedocs.io/en/latest/?badge=latest)
[![Test workflows](https://github.com/pypsa/pypsa-eur/actions/workflows/test.yaml/badge.svg)](https://github.com/pypsa/pypsa-eur/actions/workflows/test.yaml)
![Size](https://img.shields.io/github/repo-size/pypsa/pypsa-eur)
[![Zenodo PyPSA-Eur](https://zenodo.org/badge/DOI/10.5281/zenodo.3520874.svg)](https://doi.org/10.5281/zenodo.3520874)
[![Zenodo PyPSA-Eur-Sec](https://zenodo.org/badge/DOI/10.5281/zenodo.3938042.svg)](https://doi.org/10.5281/zenodo.3938042)
[![Snakemake](https://img.shields.io/badge/snakemake-≥9-brightgreen.svg?style=flat)](https://snakemake.readthedocs.io)
[![Discord](https://img.shields.io/discord/911692131440148490?logo=discord)](https://discord.gg/AnuJBk23FU)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/pypsa-eur)](https://api.reuse.software/info/github.com/pypsa/pypsa-eur)

# PyPSA-Eur: A Sector-Coupled Open Optimisation Model of the European Energy System

PyPSA-Eur is an open model dataset of the European energy system at the
transmission network level that covers the full ENTSO-E area and all energy sectors, including transport, heating, biomass, industry, and agriculture.
Besides the power grid, pipeline networks for gas, hydrogen, carbon dioxide, and liquid fuels are included.
The model is suitable both for planning studies and operational studies.
The model is built from open data using a Snakemake workflow and fully open source.
It is designed to be imported into the open-source energy system modelling framework [PyPSA](https://pypsa.org).

> [!NOTE]
> PyPSA-Eur has many contributors, with the maintenance currently led by the [Department of Digital Transformation in
> Energy Systems](https://tu.berlin/en/ensys) at the [Technical University of
> Berlin](https://www.tu.berlin).
> Previous versions were developed at the [Karlsruhe
> Institute of Technology](http://www.kit.edu/english/index.php) funded by the
> [Helmholtz Association](https://www.helmholtz.de/en/).


Among many other things, the dataset consists of:

- A power grid model based on [OpenStreetMap](https://zenodo.org/records/18619025) for voltage levels above 220kV (optional above 60kV).
- The open power plant database
  [powerplantmatching](https://github.com/PyPSA/powerplantmatching).
- Electrical demand time series from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).
- Renewable time series based on ERA5 and SARAH-3, assembled using [atlite](https://github.com/PyPSA/atlite).
- Geographical potentials for wind and solar generators based land eligibility analysis in [atlite](https://github.com/PyPSA/atlite).
- Energy balances compiled from Eurostat and JRC-IDEES datasets.

The high-voltage grid and the power plant fleet are shown in this map of the unclustered model (as of 1 January 2026):

![PyPSA-Eur Unclustered](doc/img/base.png)


For computational reasons the model is usually clustered down
to 50-250 nodes. The image below shows the electricity network and power plants clustered to NUTS2 regions:

![network diagram](doc/img/elec.png)

This diagram gives an overview of the sectors and the links between
them within each model region:

![sector diagram](doc/img/multisector_figure.png)



# Warnings

PyPSA-Eur is under active development and has several
[limitations](https://pypsa-eur.readthedocs.io/en/latest/limitations/) which
you should understand before using the model. The github repository
[issues](https://github.com/PyPSA/pypsa-eur/issues) collect known topics we are
working on (please feel free to help or make suggestions). The
[documentation](https://pypsa-eur.readthedocs.io/) remains somewhat patchy. You
can find showcases of the model's capabilities in the Joule paper [The potential
role of a hydrogen network in
Europe](https://doi.org/10.1016/j.joule.2023.06.016), another [paper in Joule
with a description of the industry
sector](https://doi.org/10.1016/j.joule.2022.04.016), or in [a 2021 presentation
at EMP-E](https://nworbmot.org/energy/brown-empe.pdf). We do not recommend to
use the full resolution network model for simulations. At high granularity the
assignment of loads and generators to the nearest network node may not be a
correct assumption, depending on the topology of the underlying distribution
grid, and local grid bottlenecks may cause unrealistic load-shedding or
generator curtailment. We recommend to cluster the network to a couple of
hundred nodes to remove these local inconsistencies. See the discussion in
Section 3.4 "Model validation" of the paper.

# Contributing and Support
We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.
-   To **discuss** with other PyPSA users, organise projects, share news, and get in touch with the community you can use the [Discord server](https://discord.gg/AnuJBk23FU).
-   For **bugs and feature requests**, please use the [PyPSA-Eur Github Issues page](https://github.com/PyPSA/pypsa-eur/issues).

# Licence

The code in PyPSA-Eur is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see [`doc/licenses.md`](doc/licenses.md).
However, different licenses and terms of use may apply to the various
input data, see [`doc/data_sources.md`](doc/data_sources.md).
