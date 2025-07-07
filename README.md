# Balancing the exploration–exploitation trade-off via multi-objective optimization for surrogate-based reliability analysis

This repository presents a multi-objective optimization (MOO) framework for balancing the exploration-exploitation trade-off in active learning of surrogate models for structural reliability analysis. Our approach computes Pareto fronts based on Gaussian process mean and uncertainty estimates, providing an optimal balance between exploration and exploitation objectives. We introduce different strategies for selecting samples from the Pareto front, including the knee point, compromise solution, and tailored-reliability method, as well as traditional U-function and EFF acquisition strategies for comparison via configuration flags.

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
├── main.py           # Entry point for active learning loop
├── active_learning/  # AcquisitionStrategy class implementation
├── limit_states/     # Benchmark limit-states
├── config/           # YAML config files for different strategies
└── README.md         # This file
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
  - Pareto front; selects samples with reliability adaptation (logistic gamma based on Pf changes).

- `default_u`

  - Strategy: U‑function (`acquisition_strategy='u'`)
  - Picks points minimizing |μ|/σ.

- `default_eff`

  - Strategy: EFF (`acquisition_strategy='eff'`)
  - Picks points maximizing the Expected Feasibility Function.
  
> **Note**: All Pareto-based strategies compute the Pareto front once per iteration. Enable the `pareto_metrics` flag in your configuration to record the full Pareto front and the selected sample. For traditional strategies, such as `u` and `eff`, the `pareto_metrics` flag computes the Pareto front, which you can then use to compare the selected sample.
---

## Example Run

```bash
python main.py --config default_moor
```

This will:

1. Load the `default_moor` config.
2. Initialize `AcquisitionStrategy('moo', moo_method='moo_reliability', N_it=..., ...)`.
3. In each iteration, pass `pf_estimate` to `strategy.get_indices(...)`.
4. Optionally at each iteration collect Pareto front and selected sample if `pareto_metrics` is enabled.

---

