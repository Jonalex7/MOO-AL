import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Postprocess_settings import (
    AGGREGATED_DIR,
    CASE_STUDIES,
    DEFAULT_STRATEGIES,
    DOE_SAMPLES,
    GROUP_2D,
    STRATEGY_COLORS,
    strategy_label,
)


DEFAULT_N_BOOTSTRAPS = 1000
DEFAULT_BOOTSTRAP_SEED = 20260312
DEFAULT_CAPTURED_LS = 5
DEFAULT_THRESHOLD_FACTOR = 1.0
DEFAULT_REQUIRED_CONSECUTIVE = 3
DEFAULT_BOOTSTRAP_SEEDS_PER_STRATEGY = 15
DEFAULT_EIER_REFERENCE_STRATEGY = "eier"
DEFAULT_EXCLUDED_GLOBAL_CASES = {"high_dimensional"}
BOOTSTRAP_DIR = AGGREGATED_DIR / "bootstrap_ranking"
SUMMARY_TXT_NAME = "bootstrap_rank_summary.txt"
LEGACY_OUTPUTS_TO_REMOVE = [
    "bootstrap_global_ranks.csv",
    "bootstrap_rank_summary.csv",
    "bootstrap_pairwise_probability.csv",
    "bootstrap_thresholds.json",
    "bootstrap_metadata.json",
    "bootstrap_pairwise_probability_heatmap.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap ranking analysis from aggregated postprocess artifacts. "
            "Thresholds are computed with EIER excluded and then applied to all "
            "strategies (including EIER)."
        )
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=DEFAULT_N_BOOTSTRAPS,
        help=f"Number of bootstrap iterations (default: {DEFAULT_N_BOOTSTRAPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--captured-ls",
        type=int,
        default=DEFAULT_CAPTURED_LS,
        help=(
            "Top-k value used to define threshold (k-th best minimum across strategies, "
            f"default: {DEFAULT_CAPTURED_LS})."
        ),
    )
    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=DEFAULT_THRESHOLD_FACTOR,
        help=(
            "Multiply threshold by this factor (>1 looser, <1 stricter, "
            f"default: {DEFAULT_THRESHOLD_FACTOR})."
        ),
    )
    parser.add_argument(
        "--required-consecutive",
        type=int,
        default=DEFAULT_REQUIRED_CONSECUTIVE,
        help=(
            "Consecutive iterations below threshold to mark hit "
            f"(default: {DEFAULT_REQUIRED_CONSECUTIVE})."
        ),
    )
    parser.add_argument(
        "--n-seeds-per-strategy",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEEDS_PER_STRATEGY,
        help=(
            "Bootstrapped runs sampled (with replacement) per strategy and case "
            f"(default: {DEFAULT_BOOTSTRAP_SEEDS_PER_STRATEGY})."
        ),
    )
    parser.add_argument(
        "--include-high-dimensional",
        action="store_true",
        help="Include high_dimensional in global average rank.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip boxplot PDF generation.",
    )
    return parser.parse_args()


def load_relative_error_dict_legacy() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    rel_path = AGGREGATED_DIR / "relative_error_dict.pkl"
    if not rel_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {rel_path}")

    with open(rel_path, "rb") as f_id:
        relative_error_dict_raw = pickle.load(f_id)

    # Keep the legacy format used in Postprocess_figures.py:
    # relative_error_dict[case][strategy][run] -> list of absolute relative error.
    relative_error_dict = {}
    for case, strategy_dict in relative_error_dict_raw.items():
        case_dict = {}
        for strategy, run_dict in strategy_dict.items():
            run_legacy = {}
            for run_name, payload in run_dict.items():
                if isinstance(payload, dict):
                    if "abs_relative_error" in payload:
                        run_legacy[run_name] = payload["abs_relative_error"]
                    elif "relative_error" in payload:
                        run_legacy[run_name] = np.abs(
                            np.asarray(payload["relative_error"], dtype=float)
                        ).tolist()
                    else:
                        run_legacy[run_name] = []
                else:
                    run_legacy[run_name] = payload
            case_dict[strategy] = run_legacy
        relative_error_dict[case] = case_dict
    return relative_error_dict


def case_max_len(case: str) -> int:
    return 200 if case in GROUP_2D else 500


def find_first_hit_index(
    rel_diff: np.ndarray,
    threshold: float,
    required_consecutive: int,
) -> Optional[int]:
    if rel_diff.size == 0:
        return None

    is_below = rel_diff <= threshold
    max_start = rel_diff.size - required_consecutive
    if max_start < 0:
        return None

    for i in range(max_start + 1):
        if np.all(is_below[i : i + required_consecutive]):
            return i
    return None


def seed_sort_key(entry: Dict[str, float]) -> tuple:
    delta = entry["delta_at_hit"]
    if delta is None or not np.isfinite(delta):
        delta = np.inf
    return (
        entry["first_hit_samples"],
        delta,
        entry["best_delta_pf"],
        entry["best_samples"],
    )


def compute_threshold_dict(
    relative_error_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
    cases: List[str],
    captured_ls: int,
    threshold_factor: float,
    excluded_from_threshold: set,
) -> Dict[str, Dict[str, object]]:
    threshold_dict = {}

    for case in cases:
        if case not in relative_error_dict:
            continue

        per_strategy_minima = []
        case_len = case_max_len(case)

        for strategy, exp_dict in relative_error_dict[case].items():
            if strategy in excluded_from_threshold or not exp_dict:
                continue

            max_len_available = max(len(arr) for arr in exp_dict.values())
            cap_len = min(max_len_available, case_len)
            rel_diff_mat = np.full((len(exp_dict), cap_len), np.nan, dtype=float)

            for row_idx, (_, rel_diff) in enumerate(sorted(exp_dict.items())):
                this_len = min(len(rel_diff), cap_len)
                rel_diff_mat[row_idx, :this_len] = rel_diff[:this_len]

            # Mirror Postprocess_figures.py threshold logic.
            median_evolution = np.median(rel_diff_mat, axis=0)
            finite_mask = np.isfinite(median_evolution)
            if not np.any(finite_mask):
                continue

            finite_indices = np.where(finite_mask)[0]
            finite_values = median_evolution[finite_mask]
            local_argmin = np.argmin(finite_values)
            it_idx = int(finite_indices[local_argmin])

            delta_pf_min = float(median_evolution[it_idx])
            n_acquired_samples = int(DOE_SAMPLES + it_idx)

            per_strategy_minima.append(
                {
                    "strategy": strategy,
                    "delta_pf_min": delta_pf_min,
                    "iteration_idx": it_idx,
                    "n_acquired_samples": n_acquired_samples,
                }
            )

        if not per_strategy_minima:
            continue

        per_strategy_minima_sorted = sorted(
            per_strategy_minima, key=lambda d: d["delta_pf_min"]
        )
        top_k = min(captured_ls, len(per_strategy_minima_sorted))
        topk = []
        for rank in range(top_k):
            entry = per_strategy_minima_sorted[rank].copy()
            entry["rank"] = rank + 1
            topk.append(entry)

        threshold_rank_index = min(captured_ls - 1, top_k - 1)
        threshold_entry = topk[threshold_rank_index]
        threshold_delta_pf = float(threshold_factor * threshold_entry["delta_pf_min"])

        threshold_dict[case] = {
            "topk_minima": topk,
            "captured_ls": captured_ls,
            "threshold_factor": threshold_factor,
            "threshold_delta_pf": threshold_delta_pf,
            "threshold_entry": threshold_entry,
        }

    return threshold_dict


def build_case_bootstrap_entries(
    case: str,
    strategies: List[str],
    relative_error_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
    threshold: float,
    rng: np.random.Generator,
    n_bootstrap_seeds_per_strategy: int,
    required_consecutive: int,
) -> List[Dict[str, object]]:
    case_seed_entries = []
    case_len = case_max_len(case)
    no_hit_value = case_len + 1

    for strategy in strategies:
        original_exp_dict = relative_error_dict.get(case, {}).get(strategy, {})
        original_keys = list(original_exp_dict.keys())
        if not original_keys:
            continue

        bootstrapped_keys = rng.choice(
            np.asarray(original_keys, dtype=object),
            size=n_bootstrap_seeds_per_strategy,
            replace=True,
        )

        for new_exp_num, original_exp_num in enumerate(bootstrapped_keys, start=1):
            rel_diff = np.asarray(original_exp_dict[original_exp_num], dtype=float)
            cap_len = min(rel_diff.size, case_len)
            rel_diff = rel_diff[:cap_len]

            finite_mask = np.isfinite(rel_diff)
            if not np.any(finite_mask):
                continue

            finite_values = rel_diff[finite_mask]
            best_delta_pf = float(np.min(finite_values))
            best_idx_local = int(np.where(rel_diff == best_delta_pf)[0][0])
            best_samples = int(DOE_SAMPLES + best_idx_local)

            first_hit_idx = find_first_hit_index(
                rel_diff,
                threshold=threshold,
                required_consecutive=required_consecutive,
            )
            if first_hit_idx is not None:
                reached = True
                first_hit_samples = int(
                    DOE_SAMPLES + first_hit_idx + (required_consecutive - 1)
                )
                delta_at_hit = float(rel_diff[first_hit_idx])
            else:
                reached = False
                first_hit_samples = no_hit_value
                delta_at_hit = None

            case_seed_entries.append(
                {
                    "case": case,
                    "strategy": strategy,
                    "exp_num": new_exp_num,
                    "global_rank": 0,
                    "reached_threshold": reached,
                    "first_hit_samples": first_hit_samples,
                    "delta_at_hit": delta_at_hit,
                    "best_delta_pf": best_delta_pf,
                    "best_samples": best_samples,
                }
            )

    return case_seed_entries


def assign_case_ranks(
    case_seed_entries: List[Dict[str, object]],
    eier_reference_strategy: str,
) -> List[Dict[str, object]]:
    pointwise_entries = [
        entry
        for entry in case_seed_entries
        if entry["strategy"] != eier_reference_strategy
    ]
    eier_entries = [
        entry
        for entry in case_seed_entries
        if entry["strategy"] == eier_reference_strategy
    ]

    pointwise_sorted = sorted(pointwise_entries, key=seed_sort_key)
    for rank_idx, entry in enumerate(pointwise_sorted, start=1):
        entry["global_rank"] = rank_idx

    for entry in eier_entries:
        current_key = seed_sort_key(entry)
        better_count = sum(seed_sort_key(other) < current_key for other in pointwise_sorted)
        entry["global_rank"] = better_count + 1

    return pointwise_sorted + sorted(eier_entries, key=seed_sort_key)


def generate_one_bootstrapped_global_ranking(
    relative_error_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
    threshold_dict: Dict[str, Dict[str, object]],
    strategies: List[str],
    cases: List[str],
    rng: np.random.Generator,
    n_bootstrap_seeds_per_strategy: int,
    required_consecutive: int,
    eier_reference_strategy: str,
) -> Dict[str, float]:
    rank_df = pd.DataFrame(index=strategies, columns=cases, dtype=float)

    for case in cases:
        if case not in relative_error_dict or case not in threshold_dict:
            continue

        threshold = float(threshold_dict[case]["threshold_delta_pf"])
        case_entries = build_case_bootstrap_entries(
            case=case,
            strategies=strategies,
            relative_error_dict=relative_error_dict,
            threshold=threshold,
            rng=rng,
            n_bootstrap_seeds_per_strategy=n_bootstrap_seeds_per_strategy,
            required_consecutive=required_consecutive,
        )
        ranked_entries = assign_case_ranks(
            case_seed_entries=case_entries,
            eier_reference_strategy=eier_reference_strategy,
        )

        ranks_by_strategy = {}
        for row in ranked_entries:
            ranks_by_strategy.setdefault(row["strategy"], []).append(row["global_rank"])

        for strategy, ranks in ranks_by_strategy.items():
            rank_df.loc[strategy, case] = float(np.mean(ranks))

    return rank_df.mean(axis=1, skipna=True).to_dict()


def compute_pairwise_probability_matrix(
    bootstrap_ranks_df: pd.DataFrame,
    strategies: List[str],
) -> pd.DataFrame:
    probability_matrix = pd.DataFrame(0.0, index=strategies, columns=strategies)
    for strat_a in strategies:
        for strat_b in strategies:
            if strat_a == strat_b:
                continue

            valid = bootstrap_ranks_df[[strat_a, strat_b]].dropna()
            if valid.empty:
                continue

            probability = float((valid[strat_a] < valid[strat_b]).mean())
            probability_matrix.loc[strat_a, strat_b] = probability

    return probability_matrix


def _format_rank_summary_table(rank_summary: pd.DataFrame) -> str:
    printable = rank_summary.copy()
    printable["strategy"] = printable["strategy"].map(strategy_label)
    printable = printable.rename(
        columns={
            "strategy": "Strategy",
            "mean_global_rank": "Mean Rank",
            "median_global_rank": "Median Rank",
            "p2_5_global_rank": "P2.5 Rank",
            "p97_5_global_rank": "P97.5 Rank",
            "n_valid_bootstraps": "Valid N",
        }
    )
    return printable.to_string(
        index=False,
        formatters={
            "Mean Rank": "{:.3f}".format,
            "Median Rank": "{:.3f}".format,
            "P2.5 Rank": "{:.3f}".format,
            "P97.5 Rank": "{:.3f}".format,
            "Valid N": "{:d}".format,
        },
    )


def save_readable_summary_txt(
    output_path: Path,
    rank_summary: pd.DataFrame,
    probability_sorted: pd.DataFrame,
    threshold_dict: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("Bootstrap Ranking Summary")
    lines.append("=" * 80)
    lines.append(f"N_BOOTSTRAPS         : {args.n_bootstraps}")
    lines.append(f"Random seed          : {args.seed}")
    lines.append(f"Captured top-k (ls)  : {args.captured_ls}")
    lines.append(f"Threshold factor     : {args.threshold_factor}")
    lines.append(f"Required consecutive : {args.required_consecutive}")
    lines.append(f"Bootstrap seeds/strat: {args.n_seeds_per_strategy}")
    lines.append("")
    lines.append("Thresholds by case")
    lines.append("-" * 80)
    for case, payload in threshold_dict.items():
        threshold = payload["threshold_delta_pf"]
        source = payload["threshold_entry"]["strategy"]
        lines.append(
            f"threshold[{case}] = {threshold:.3e} "
            f"(defined by {strategy_label(source)})"
        )
    lines.append("")
    lines.append("Global rank table")
    lines.append("-" * 80)
    lines.append(_format_rank_summary_table(rank_summary))
    lines.append("")
    lines.append("Pairwise probability matrix")
    lines.append("-" * 80)
    prob_print = probability_sorted.round(3).copy()
    prob_print.index = [strategy_label(s) for s in prob_print.index]
    prob_print.columns = [strategy_label(s) for s in prob_print.columns]
    lines.append(prob_print.to_string())
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f_id:
        f_id.write("\n".join(lines))


def plot_bootstrap_rank_boxplot(
    bootstrap_ranks_df: pd.DataFrame,
    ordered_strategies: List[str],
    output_path: Path,
    n_bootstraps: int,
) -> None:
    import matplotlib.pyplot as plt

    data = [bootstrap_ranks_df[s].dropna().to_numpy() for s in ordered_strategies]
    labels = [strategy_label(s) for s in ordered_strategies]
    colors = [STRATEGY_COLORS.get(s, "#808080") for s in ordered_strategies]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.65,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.1)

    ax.set_title(f"Bootstrapped Global Average Rank Distribution ({n_bootstraps} samples)")
    ax.set_ylabel("Global Average Rank (lower is better)")
    ax.set_xlabel("Acquisition Strategy")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def remove_legacy_outputs(output_dir: Path) -> None:
    for name in LEGACY_OUTPUTS_TO_REMOVE:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()

    relative_error_dict = load_relative_error_dict_legacy()
    available_cases = [c for c in CASE_STUDIES if c in relative_error_dict]
    if not available_cases:
        raise RuntimeError(
            "No case data found in relative_error_dict.pkl. "
            "Run Postprocess_output_files.py first."
        )

    strategies = [s for s in DEFAULT_STRATEGIES if any(s in relative_error_dict[c] for c in available_cases)]
    if not strategies:
        raise RuntimeError(
            "No strategy data found in relative_error_dict.pkl for configured case studies."
        )

    if not args.include_high_dimensional:
        cases = [c for c in available_cases if c not in DEFAULT_EXCLUDED_GLOBAL_CASES]
    else:
        cases = available_cases

    threshold_dict = compute_threshold_dict(
        relative_error_dict=relative_error_dict,
        cases=cases,
        captured_ls=args.captured_ls,
        threshold_factor=args.threshold_factor,
        excluded_from_threshold={DEFAULT_EIER_REFERENCE_STRATEGY},
    )

    missing_threshold_cases = [c for c in cases if c not in threshold_dict]
    if missing_threshold_cases:
        print(f"[warn] no threshold computed for cases: {missing_threshold_cases}")
    cases_for_bootstrap = [c for c in cases if c in threshold_dict]
    if not cases_for_bootstrap:
        raise RuntimeError("No valid cases available for bootstrap ranking after threshold computation.")

    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    remove_legacy_outputs(BOOTSTRAP_DIR)

    print("[start] bootstrap ranking analysis")
    print(f"  aggregated dir       : {AGGREGATED_DIR}")
    print(f"  output dir           : {BOOTSTRAP_DIR}")
    print(f"  n_bootstraps         : {args.n_bootstraps}")
    print(f"  random seed          : {args.seed}")
    print(f"  captured_ls          : {args.captured_ls}")
    print(f"  threshold_factor     : {args.threshold_factor}")
    print(f"  required_consecutive : {args.required_consecutive}")
    print(f"  n_seeds_per_strategy : {args.n_seeds_per_strategy}")
    print(f"  cases                : {cases_for_bootstrap}")
    print(f"  strategies           : {strategies}")
    print(f"  excluded threshold   : [{DEFAULT_EIER_REFERENCE_STRATEGY}]")
    print("")

    for case in cases_for_bootstrap:
        threshold = threshold_dict[case]["threshold_delta_pf"]
        source = threshold_dict[case]["threshold_entry"]["strategy"]
        print(
            f"  threshold[{case}] = {threshold:.3e} "
            f"(defined by {strategy_label(source)})"
        )

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    for b in range(args.n_bootstraps):
        if (b + 1) % 100 == 0:
            print(f"  ... iteration {b + 1}/{args.n_bootstraps}")

        global_ranks = generate_one_bootstrapped_global_ranking(
            relative_error_dict=relative_error_dict,
            threshold_dict=threshold_dict,
            strategies=strategies,
            cases=cases_for_bootstrap,
            rng=rng,
            n_bootstrap_seeds_per_strategy=args.n_seeds_per_strategy,
            required_consecutive=args.required_consecutive,
            eier_reference_strategy=DEFAULT_EIER_REFERENCE_STRATEGY,
        )
        bootstrap_rows.append(global_ranks)

    bootstrap_ranks_df = pd.DataFrame(bootstrap_rows, columns=strategies)
    rank_summary = pd.DataFrame(
        {
            "strategy": strategies,
            "mean_global_rank": [float(np.nanmean(bootstrap_ranks_df[s])) for s in strategies],
            "median_global_rank": [float(np.nanmedian(bootstrap_ranks_df[s])) for s in strategies],
            "p2_5_global_rank": [float(np.nanpercentile(bootstrap_ranks_df[s], 2.5)) for s in strategies],
            "p97_5_global_rank": [float(np.nanpercentile(bootstrap_ranks_df[s], 97.5)) for s in strategies],
            "n_valid_bootstraps": [int(bootstrap_ranks_df[s].notna().sum()) for s in strategies],
        }
    ).sort_values("mean_global_rank", ascending=True)
    probability_matrix = compute_pairwise_probability_matrix(
        bootstrap_ranks_df=bootstrap_ranks_df,
        strategies=strategies,
    )

    ordered_strategies = rank_summary["strategy"].tolist()
    probability_sorted = probability_matrix.loc[ordered_strategies, ordered_strategies]

    print("")
    print("--- Pairwise Probability Matrix: P(Row Strategy is Better than Column Strategy) ---")
    probability_print = probability_sorted.round(3).copy()
    probability_print.index = [strategy_label(s) for s in probability_print.index]
    probability_print.columns = [strategy_label(s) for s in probability_print.columns]
    print(probability_print.to_string())

    summary_txt_path = BOOTSTRAP_DIR / SUMMARY_TXT_NAME
    save_readable_summary_txt(
        output_path=summary_txt_path,
        rank_summary=rank_summary,
        probability_sorted=probability_sorted,
        threshold_dict=threshold_dict,
        args=args,
    )

    if not args.no_plots:
        plot_bootstrap_rank_boxplot(
            bootstrap_ranks_df=bootstrap_ranks_df,
            ordered_strategies=ordered_strategies,
            output_path=BOOTSTRAP_DIR / "bootstrap_rank_boxplot.pdf",
            n_bootstraps=args.n_bootstraps,
        )

    print("")
    print("[done] bootstrap artifacts written:")
    print(f"  - {summary_txt_path}")
    if not args.no_plots:
        print(f"  - {BOOTSTRAP_DIR / 'bootstrap_rank_boxplot.pdf'}")


if __name__ == "__main__":
    main()
