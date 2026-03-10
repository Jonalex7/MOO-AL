from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RESULTS_DIR = REPO_ROOT / "notebooks" / "results_tracking"
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


STRATEGY_DIR_MAP = {
    "moo_eps_ew": "moo_eps_lw",
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
    "eier": "#000000",
    "eff": "#d62728",
    "u": "#9467bd",
    "erf": "#17becf",
    "reif": "#432ca0",
    "reif2": "#e377c2",
    "portfolio": "#7f7f7f",
}


DOE_SAMPLES = 10
MAX_LEN_BY_CASE = {case: (200 if case in GROUP_2D else 500) for case in CASE_STUDIES}


def strategy_to_dir(strategy: str) -> str:
    return STRATEGY_DIR_MAP.get(strategy, strategy)


def strategy_label(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


def case_title(case: str) -> str:
    return CASE_TITLES.get(case, case)

