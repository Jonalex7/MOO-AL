# Balancing the exploration–exploitation trade-off via multi-objective optimization for surrogate-based reliability analysis

This repository provides an active-learning framework for surrogate-based structural reliability analysis, centered on multi-objective optimization (MOO) Pareto-based sampling. In addition to the proposed MOO variants, the codebase also supports conventional baselines for direct comparison within the same training/evaluation pipeline.

Supported acquisition families:

- MOO-based: `knee`, `compromise`, `reliability`, `linear_decay`.
- Non-MOO: `u`, `eff`, `erf`, `reif`, `reif2`, `portfolio`, and `eier`.

---

## Prerequisites

The required Python packages are listed in `requirements.txt`:

```text
numpy==1.26.1
scikit-learn==1.5.1
```

Ensure you have Python 3.8+ installed.

### Install dependencies

Use the provided requirements file to install all necessary packages:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── main.py              # Entry point for active learning loop
├── active_learning/     # AcquisitionStrategy implementation
├── limit_states/        # Benchmark limit-state functions
├── config/              # YAML configs for MOO and non-MOO strategies
├── results/             # Raw run outputs by case/strategy and aggregated artifacts
│   └── _aggregated/     # Campaign-level postprocessed dictionaries/figures/tables
├── postprocess/         # Postprocessing scripts (aggregation, figures, bootstrap ranking)
├── notebooks/           # Exploratory notebooks
└── README.md            # This file
```

Typical workflow:

1. Run an experimental campaign (usually repeated runs across one or more configs/seeds), which writes raw outputs in `results/<case>/<method>/<run>/`.
2. Aggregate campaign outputs into `results/_aggregated/`.
3. Generate report-ready figures/tables from `postprocess/`.

---

## Usage

Run the active learning loop by specifying one of the predefined configurations:

```bash
python main.py --config <AL_CONFIG>
```

Optional: set `save_model_gp: True` in the config (or pass `--save_model_gp true`) to store GP checkpoints in each run folder. Default is `False`.

### Available Configurations

- `default_mook`

  - Strategy: MOO‑knee (`acquisition_strategy='moo'`, `moo_method='knee'`)
  - Pareto front; selects the knee point.

- `default_mooc`

  - Strategy: MOO‑compromise (`acquisition_strategy='moo'`, `moo_method='compromise'`)
  - Pareto front; selects the compromise (closest to ideal) point.

- `default_moor`

  - Strategy: MOO‑reliability (`acquisition_strategy='moo'`, `moo_method='reliability'`)
  - Pareto front; selects samples with reliability adaptation (logistic gamma based on Pf changes).

- `default_moold`

  - Strategy: MOO‑linear‑decay (`acquisition_strategy='moo'`, `moo_method='linear_decay'`)
  - Pareto front; linear decay preference via Euclidean-compromise scalarization.

- `default_u`

  - Strategy: U‑function (`acquisition_strategy='u'`)
  - Picks points minimizing |μ|/σ.

- `default_eff`

  - Strategy: EFF (`acquisition_strategy='eff'`)
  - Picks points maximizing the Expected Feasibility Function.

- `default_erf`

  - Strategy: ERF (`acquisition_strategy='erf'`)
  - Picks points maximizing Expected Risk Function.

- `default_reif`

  - Strategy: REIF/REIF2 family (`acquisition_strategy='reif2'` in current default file)
  - Risk-based feasibility criterion.

- `default_portfolio`

  - Strategy: Portfolio hedge (`acquisition_strategy='portfolio'`)
  - Adaptive combination of non-MOO pointwise arms.

- `default_eier`

  - Strategy: EIER (`acquisition_strategy='eier'`)
  - One-step expected information gain style look-ahead selection.
  
> **Note**: For `al_strategy='moo'`, the Pareto front is part of the selection step. For non-MOO strategies, enabling `pareto_metrics: True` stores Pareto diagnostics for comparison/reporting.
---

## Example Run

```bash
python main.py --config default_moor
```

This will:

1. Load the `default_moor` config.
2. Initialize `AcquisitionStrategy('moo', moo_method='reliability', N_it=..., ...)`.
3. In each iteration, pass `pf_estimate` to `strategy.get_indices(...)`.
4. Optionally at each iteration collect Pareto front and selected sample if `pareto_metrics` is enabled.

## Postprocessing

After a campaign (including repeated runs), use:

```bash
python postprocess/output_files.py
python postprocess/figures.py
python postprocess/bootstrap_ranking.py --n-bootstraps 1000
```

These scripts read campaign outputs and write aggregated report artifacts under `results/_aggregated/`.

---
