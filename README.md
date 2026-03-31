# Balancing the exploration-exploitation trade-off via multi-objective optimization for surrogate-based reliability analysis

This repository provides an active-learning framework for surrogate-based structural reliability analysis, centered on multi-objective optimization (MOO) Pareto-based sampling. In addition to the proposed MOO variants, the codebase also supports conventional baselines for direct comparison within the same training and evaluation pipeline.

Supported acquisition families:

- MOO-based: `knee`, `compromise`, `reliability`, `linear_decay`
- Non-MOO: `u`, `eff`, `erf`, `reif`, `reif2`, `portfolio`, `eier`

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

```text
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

1. Run an experimental campaign, usually repeated runs across one or more configs and seeds, which writes raw outputs in `results/<case>/<method>/<run>/`.
2. Aggregate campaign outputs into `results/_aggregated/`.
3. Generate report-ready figures and tables from `postprocess/`.

---

## Usage

Run the active learning loop by specifying one of the predefined configurations:

```bash
python main.py --config <AL_CONFIG>
```

Optional: set `save_model_gp: True` in the config, or pass `--save_model_gp true`, to store GP checkpoints in each run folder. The default is `False`.

### Available Configurations

All default config files share the same general experiment keys, such as:
`case_study`, `al_batch`, `total_samples`, `passive_samples`, `n_mcs_pool`,
`seed`, `n_mcs_pf`, `save_interval`, `save_model_gp`, `pareto_metrics`, and
`wandb_online`.

The list below only highlights the keys that are specific to the corresponding
acquisition method.

- `default_mook`

  - Strategy: MOO-knee (`al_strategy='moo'`, `moo_method='knee'`)
  - Pareto front; selects the knee point.

- `default_mooc`

  - Strategy: MOO-compromise (`al_strategy='moo'`, `moo_method='compromise'`)
  - Pareto front; selects the compromise point, i.e. the point closest to the ideal objective pair.

- `default_moor`

  - Strategy: MOO-reliability (`al_strategy='moo'`, `moo_method='reliability'`)
  - Method-specific keys:
    - `N_it`
    - `delta_p0`
    - `k_balance`
  - Pareto front; selects samples with reliability estimate convergence through the gamma update.

- `default_moold`

  - Strategy: MOO-linear-decay (`al_strategy='moo'`, `moo_method='linear_decay'`)
  - Method-specific keys:
    - `eps_start`
    - `eps_end`
    - `eps_T`
  - Pareto front; linear-decay preference via Euclidean-compromise scalarization.

- `default_u`

  - Strategy: U-function (`al_strategy='u'`)
  - Selects points minimizing `|mu| / sigma`.

- `default_eff`

  - Strategy: EFF (`al_strategy='eff'`)
  - Method-specific keys:
    - `eff_constant`
  - Selects points maximizing the Expected Feasibility Function.

- `default_erf`

  - Strategy: ERF (`al_strategy='erf'`)
  - Selects points maximizing the Expected Risk Function.

- `default_reif`

  - Strategy: REIF/REIF2 family (`al_strategy='reif2'` in the current default file)
  - Method-specific keys:
    - `reif_w`
  - To run the original REIF instead of REIF2, change `al_strategy` from `reif2` to `reif`.
  - Risk-based feasibility criterion.

- `default_portfolio`

  - Strategy: Portfolio hedge (`al_strategy='portfolio'`)
  - Method-specific keys:
    - `portfolio_lambda`
    - `portfolio_delta`
  - Adaptive combination of non-MOO pointwise arms.

- `default_eier`

  - Strategy: EIER (`al_strategy='eier'`)
  - Method-specific keys:
    - `n_mcs_eier_int`
    - `local_mis_topk`
    - `n_g_pf`
  - One-step expected information gain style look-ahead selection.

> **Note**: For `al_strategy='moo'`, the Pareto front is part of the selection step. For non-MOO strategies, enabling `pareto_metrics: True` stores Pareto diagnostics for comparison and reporting.

---

## Example Run

```bash
python main.py --config default_moor
```

This will:

1. Load the `default_moor` config.
2. Initialize `AcquisitionStrategy('moo', moo_method='reliability', N_it=..., ...)`.
3. In each iteration, pass `pf_estimate` to `strategy.get_indices(...)`.
4. Optionally collect Pareto-front diagnostics if `pareto_metrics` is enabled.

## Postprocessing

After a campaign, including repeated runs, use:

```bash
python postprocess/output_files.py
python postprocess/figures.py
python postprocess/bootstrap_ranking.py --n-bootstraps 1000
```

These scripts read campaign outputs and write aggregated report artifacts under `results/_aggregated/`.

---
