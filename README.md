# Balancing the exploration–exploitation trade-off via multi-objective optimization for surrogate-based reliability analysis

This repository presents a multi-objective optimization framework for balancing the exploration-exploitation trade-off in active learning of surrogate models for structural reliability analysis. Our approach computes Pareto fronts based on Gaussian process mean and uncertainty estimates, providing an optimal balance between exploration and exploitation objectives. We introduce different strategies for selecting samples from the Pareto front, including the knee point, compromise solution, and tailored-reliability method, as well as traditional U-function and EFF acquisition strategies for comparison via configuration flags.

---

## Prerequisites

The required Python packages are listed in `requirements.txt`:

```text
numpy==1.26.1
torch==2.4.0
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
├── main.py          # Entry point for active learning loop
├── acquisition.py   # AcquisitionStrategy class implementation
├── config/          # YAML config files for different strategies
└── README.md        # This file
```

---

## Usage

Run the active learning loop by specifying one of the predefined configurations:

```bash
python main.py --config <AL_CONFIG>
```

### Available Configurations

- `default_mook`

  - Strategy: MOO‑knee (`acquisition_strategy='moo'`, `moo_method='knee'`)
  - Pareto front; selects the knee point.

- `default_mooc`

  - Strategy: MOO‑compromise (`acquisition_strategy='moo'`, `moo_method='compromise'`)
  - Pareto front; selects the compromise (closest to ideal) point.

- `default_moor`

  - Strategy: MOO‑reliability (`acquisition_strategy='moo'`, `moo_method='moo_reliability'`)
  - Pareto front with reliability adaptation (logistic gamma based on Pf changes).

- `default_u`

  - Strategy: U‑function (`acquisition_strategy='u'`)
  - Picks points minimizing |μ|/σ.

- `default_eff`

  - Strategy: EFF (`acquisition_strategy='eff'`)
  - Picks points maximizing the Expected Feasibility Function.
  
> **Note**: All Pareto‑based strategies compute the Pareto front once per iteration. You can enable the `pareto_metrics` flag in your config to record the full Pareto front and the selected sample.

---

## Example Run

```bash
python main.py --config default_moor
```

This will:

1. Load the `default_moor` config.
2. Initialize `AcquisitionStrategy('moo', moo_method='moo_reliability', N_it=..., ...)`.
3. In each iteration, pass `pf_estimate` to `strategy.get_indices(...)`.
4. Optionally collect Pareto metrics if `pareto_metrics` is enabled.

---

