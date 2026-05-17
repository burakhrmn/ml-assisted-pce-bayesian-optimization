# Project Structure

## `data/`

Contains project data. The raw CSV is stored in `data/raw/data.csv` and is not manually edited by the workflow.

`data/raw/data_example.csv` is included as an anonymized example input that demonstrates the expected raw data format.

`data/raw/data.csv` is ignored by Git and should be kept local unless intentionally shared.

## `src/`

Contains reusable Python modules for configuration, data loading, grouping, visualization, model comparison, Bayesian Optimization, and report generation.

## `outputs/tables/`

Contains generated CSV and text outputs, including filtered data, grouped condition summaries, model comparison metrics, candidate predictions, recommendation tables, diagnostics, and the curated experimental plan.

## `outputs/figures/`

Contains generated exploratory, model comparison, Bayesian Optimization, and recommendation figures.

## `main.py`

Runs the full reproducible workflow from raw data loading through final report generation.

## `README.md`

Provides the project overview, scientific interpretation, installation instructions, run instructions, output descriptions, and limitations.

## `requirements.txt`

Lists the Python packages required to run the project.
