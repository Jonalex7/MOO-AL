from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS_DIR = REPO_ROOT / "results"
AGGREGATED_DIR = BASE_RESULTS_DIR / "_aggregated"
FIGURES_DIR = BASE_RESULTS_DIR / "_figures"


GROUP_2D = ["four_branch_6", "four_branch_7", "hat", "himmelblau"]
GROUP_HD = ["nonlinear_oscillator", "2dof_oscillator", "high_dimensional"]
CASE_STUDIES = GROUP_2D + GROUP_HD


CASE_TITLES = {
    "four_branch_6": r"Four-branch, $k=6$",
    "four_branch_7": r"Four-branch, $k=7$",
    "hat": "Hat",
    "himmelblau": "Himmelblau",
    "nonlinear_oscillator": "Nonlinear oscillator",
    "2dof_oscillator": "2-DOF oscillator",
    "high_dimensional": "High-dimensional",
}

REAL_PF_VALUES = {
    "four_branch_6": 0.004458488011732697,
    "four_branch_7": 0.0022232679883018138,
    "hat": 0.00038667799963150175,
    "himmelblau": 1.65e-4,
    "nonlinear_oscillator": 0.0286178,
    "2dof_oscillator": 0.0047598,
    "high_dimensional": 0.0019820,
}


DEFAULT_STRATEGIES = [
    "moo_reliability",
    "moo_knee",
    "moo_compromise",
    "moo_eps_ew",
    "eier",
    "eff",
    "u",
    "erf",
    "reif",
    "reif2",
    "portfolio",
]


# Map strategy keys to preferred results folders.
# For linear-decay MOO runs, new folders use `moo_linear_decay`,
# while legacy artifacts may still exist as `moo_eps_ew`.
STRATEGY_DIR_MAP = {
    "moo_eps_ew": "moo_linear_decay",
}


STRATEGY_LABELS = {
    "moo_reliability": "MOO-R",
    "moo_knee": "MOO-K",
    "moo_compromise": "MOO-C",
    "moo_eps_ew": "MOO-LD",
    "eier": "EIER",
    "eff": "EFF",
    "u": "U",
    "erf": "ERF",
    "reif": "REIF",
    "reif2": "REIF2",
    "portfolio": "Portfolio",
}


STRATEGY_COLORS = {
    "moo_reliability": "#1f77b4",
    "moo_knee": "#ff7f0e",
    "moo_compromise": "#2ca02c",
    "moo_eps_ew": "#8c564b",
    "eier": "#1f5f6b",
    "eff": "#d62728",
    "u": "#9467bd",
    "erf": "#17becf",
    "reif": "#432ca0",
    "reif2": "#e377c2",
    "portfolio": "#7f7f7f",
}


DOE_SAMPLES = 10
MAX_LEN_BY_CASE = {case: (202 if case in GROUP_2D else 502) for case in CASE_STUDIES}

# Threshold-detection defaults shared by the sample-efficiency and posterior-CoV
# postprocess steps.
CAPTURED_LS = 5
THRESHOLD_FACTOR = 1.0
REQUIRED_CONSECUTIVE = 3
EIER_REFERENCE_STRATEGY = "eier"

# Posterior-estimation defaults used when old runs do not store these values in
# config.json or output.json.
DEFAULT_POST_N_G_PF = 1000
DEFAULT_POST_N_PF_POST_POOL = 10000000
DEFAULT_POST_PF_POST_BATCH_SIZE = 500
DEFAULT_POST_PREDICT_BATCH_SIZE = 10000
DEFAULT_POST_PREDICT_N_JOBS = -1

# Aggregated artifacts produced by the figures and posterior-CoV campaigns.
THRESHOLD_DICT_NAME = "thresholds_by_case.json"
THRESHOLD_HITS_TABLE_NAME = "threshold_hits_per_seed.tsv"
STRATEGY_RANKINGS_TABLE_NAME = "strategy_rankings_by_case.tsv"
PF_POST_COV_TABLE_NAME = "pf_post_cov_at_threshold_per_seed.tsv"
PF_POST_COV_SUMMARY_NAME = "pf_post_cov_at_threshold_summary.txt"
PF_POST_SAMPLES_DIRNAME = "pf_post_samples"


def strategy_to_dir(strategy: str) -> str:
    return STRATEGY_DIR_MAP.get(strategy, strategy)


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def case_title(case: str) -> str:
    return CASE_TITLES.get(case, case)
