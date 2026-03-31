import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings import PF_POST_COV_TABLE_NAME, REPO_ROOT as SETTINGS_REPO_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one saved posterior Pf sample distribution from the aggregated "
            "pf_post_cov campaign table and save a quick histogram plot."
        )
    )
    parser.add_argument(
        "--results-folder",
        "--results_folder",
        dest="results_folder",
        default="results_2ndrev",
        help="Folder under the repository root containing the aggregated campaign outputs.",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Optional direct path to a saved pf_post_samples .npz artifact.",
    )
    parser.add_argument("--case-study", required=True, help="Case-study key, e.g. four_branch_6.")
    parser.add_argument("--strategy", required=True, help="Strategy key, e.g. eier.")
    parser.add_argument("--run", required=True, help="Run folder name.")
    parser.add_argument(
        "--n-g-pf",
        type=int,
        default=None,
        help="Optional n_g_pf filter when multiple saved distributions exist for the same run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output path for the histogram figure.",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = SETTINGS_REPO_ROOT / path
    return path.resolve()


def load_rows(table_path: Path):
    with open(table_path, "r", encoding="utf-8", newline="") as f_id:
        return list(csv.DictReader(f_id, delimiter="\t"))


def parse_sample_path_metadata(sample_path: Path, case: str, strategy: str, run_name: str, n_g_pf):
    pattern = re.compile(
        r"__step(?P<step>\d+)__train(?P<train>\d+)__(?P<estimator>[^_]+)__ng(?P<ng>\d+)__pool(?P<pool>\d+)\.npz$"
    )
    match = pattern.search(sample_path.name)
    if match is None:
        raise ValueError(f"Could not parse metadata from sample artifact name: {sample_path.name}")
    ng_value = int(match.group("ng"))
    if n_g_pf is not None and int(n_g_pf) != ng_value:
        raise ValueError(f"Sample artifact n_g_pf={ng_value} does not match requested --n-g-pf={n_g_pf}.")
    return {
        "case": case,
        "strategy": strategy,
        "run": run_name,
        "n_g_pf": str(ng_value),
        "n_pf_post_pool": match.group("pool"),
        "evaluation_step": match.group("step"),
        "evaluation_train_size": match.group("train"),
        "posterior_estimator": match.group("estimator"),
        "pf_post_samples_path": str(sample_path),
        "pf_model_replayed": None,
    }


def find_sample_path(aggregated_dir: Path, case: str, strategy: str, run_name: str, n_g_pf):
    samples_dir = aggregated_dir / "pf_post_samples" / case / strategy
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {samples_dir}")
    candidates = sorted(samples_dir.glob(f"{run_name}__*.npz"))
    if n_g_pf is not None:
        candidates = [path for path in candidates if f"__ng{int(n_g_pf)}__" in path.name]
    if not candidates:
        raise FileNotFoundError("No matching sample artifact was found on disk.")
    if len(candidates) > 1:
        raise ValueError(
            "Multiple matching sample artifacts were found. Specify --n-g-pf or use --sample-path.\n"
            + "\n".join(str(path) for path in candidates)
        )
    return candidates[0]


def select_row(rows, aggregated_dir: Path, case, strategy, run_name, n_g_pf, sample_path=None):
    if sample_path is not None:
        return parse_sample_path_metadata(sample_path, case, strategy, run_name, n_g_pf)

    matches = [
        row
        for row in rows
        if row.get("case") == case and row.get("strategy") == strategy and row.get("run") == run_name
    ]
    if n_g_pf is not None:
        matches = [row for row in matches if int(row.get("n_g_pf") or 0) == int(n_g_pf)]

    matches = [row for row in matches if row.get("pf_post_samples_path")]
    if not matches:
        sample_path = find_sample_path(aggregated_dir, case, strategy, run_name, n_g_pf)
        return parse_sample_path_metadata(sample_path, case, strategy, run_name, n_g_pf)
    if len(matches) > 1:
        available = sorted({row.get("n_g_pf") for row in matches})
        raise ValueError(
            "Multiple matching rows were found. Specify --n-g-pf. "
            f"Available n_g_pf values: {available}"
        )
    return matches[0]


def summarize(samples: np.ndarray):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    cov = float(std / max(mean, 1e-16))
    q2_5, q50, q97_5 = np.quantile(samples, [0.025, 0.5, 0.975])
    return {
        "count": int(samples.size),
        "mean": mean,
        "std": std,
        "cov": cov,
        "q2_5": float(q2_5),
        "median": float(q50),
        "q97_5": float(q97_5),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
    }


def default_output_path(aggregated_dir: Path, row):
    output_dir = aggregated_dir / "pf_post_sample_plots" / row["case"] / row["strategy"]
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"ng{row.get('n_g_pf') or 'na'}"
    return output_dir / f"{row['run']}__{suffix}.png"


def make_plot(samples: np.ndarray, row, summary, output_path: Path):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    bins = min(40, max(12, samples.size // 20))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(samples, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.axvline(summary["mean"], color="#D62728", linewidth=2.0, label="Mean")
    ax.axvline(summary["median"], color="black", linewidth=1.8, linestyle="--", label="Median")
    ax.axvline(summary["q2_5"], color="#2CA02C", linewidth=1.5, linestyle=":", label="CI95")
    ax.axvline(summary["q97_5"], color="#2CA02C", linewidth=1.5, linestyle=":")

    pf_model_replayed = row.get("pf_model_replayed")
    if pf_model_replayed not in (None, "", "None"):
        ax.axvline(float(pf_model_replayed), color="#9467BD", linewidth=1.8, label="Pf_model")

    ax.set_title(
        f"{row['case']} / {row['strategy']} / {row['run']}\n"
        f"train_size={row['evaluation_train_size']} | n_g_pf={row['n_g_pf']} | n_pool={row['n_pf_post_pool']}"
    )
    ax.set_xlabel("Posterior Pf samples")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    base_results_dir = resolve_path(args.results_folder)
    aggregated_dir = base_results_dir / "_aggregated"
    table_path = aggregated_dir / PF_POST_COV_TABLE_NAME
    rows = load_rows(table_path) if table_path.is_file() else []
    sample_path_arg = None if args.sample_path is None else resolve_path(args.sample_path)
    row = select_row(rows, aggregated_dir, args.case_study, args.strategy, args.run, args.n_g_pf, sample_path_arg)
    sample_path = Path(row["pf_post_samples_path"])
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing saved pf_post_samples artifact: {sample_path}")

    with np.load(sample_path) as data:
        samples = np.asarray(data["pf_samples"], dtype=np.float64)

    summary = summarize(samples)
    output_path = resolve_path(args.output) if args.output else default_output_path(aggregated_dir, row)
    make_plot(samples, row, summary, output_path)

    print(f"Sample artifact           : {sample_path}")
    print(f"Plot saved               : {output_path}")
    print(f"Sample count             : {summary['count']}")
    print(f"Mean                     : {summary['mean']:.12e}")
    print(f"Std                      : {summary['std']:.12e}")
    print(f"CoV                      : {summary['cov']:.12e}")
    print(f"Median                   : {summary['median']:.12e}")
    print(f"CI95                     : [{summary['q2_5']:.12e}, {summary['q97_5']:.12e}]")
    print(f"Min / Max                : [{summary['min']:.12e}, {summary['max']:.12e}]")


if __name__ == "__main__":
    main()
