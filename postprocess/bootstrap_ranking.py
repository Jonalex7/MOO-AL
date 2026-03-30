import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from settings import (
    AGGREGATED_DIR,
    BASE_RESULTS_DIR,
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
DEFAULT_DOMINANCE_THRESHOLD = 0.9
DEFAULT_EIER_REFERENCE_STRATEGY = "eier"
DEFAULT_EXCLUDED_GLOBAL_CASES = {"high_dimensional"}
BOOTSTRAP_DIR = AGGREGATED_DIR / "bootstrap_ranking"
SUMMARY_TXT_NAME = "bootstrap_rank_summary.txt"
RANK_SUMMARY_CSV_NAME = "bootstrap_rank_summary.csv"
RANK_POSITIONS_CSV_NAME = "bootstrap_rank_positions.csv"
RANK_POSITION_SUMMARY_CSV_NAME = "bootstrap_rank_position_summary.csv"
PAIRWISE_PROBABILITY_CSV_NAME = "bootstrap_pairwise_probability.csv"
PAIRWISE_RELATION_CSV_NAME = "bootstrap_pairwise_relation.csv"
DOMINANCE_TIERS_CSV_NAME = "bootstrap_dominance_tiers.csv"
DOMINANCE_BAR_PDF_NAME = "bootstrap_dominance_stacked_bar.pdf"
BOOTSTRAP_METADATA_NAME = "bootstrap_metadata.json"
LEGACY_OUTPUTS_TO_REMOVE = [
    "bootstrap_global_ranks.csv",
    "bootstrap_rank_summary.csv",
    "bootstrap_pairwise_probability.csv",
    "bootstrap_thresholds.json",
    "bootstrap_metadata.json",
    "bootstrap_pairwise_probability_heatmap.pdf",
]


def _default_results_folder() -> str:
    return str(BASE_RESULTS_DIR.relative_to(Path.cwd()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap ranking analysis from aggregated postprocess artifacts. "
            "Thresholds are computed with EIER excluded and then applied to all "
            "strategies (including EIER)."
        )
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default=_default_results_folder(),
        help=(
            "Results root containing the per-case directories and its own "
            f"_aggregated folder (default: {_default_results_folder()})."
        ),
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
        "--dominance-threshold",
        type=float,
        default=DEFAULT_DOMINANCE_THRESHOLD,
        help=(
            "Threshold for strong dominance in pairwise probability matrix "
            f"(default: {DEFAULT_DOMINANCE_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--reuse-existing-probability",
        action="store_true",
        help=(
            "Reuse the saved pairwise probability matrix and rank summary from the "
            "existing bootstrap output folder, and only recompute dominance/tier "
            "artifacts for the requested dominance threshold."
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


def resolve_results_dirs(results_folder: str) -> Tuple[Path, Path, Path]:
    base_results_dir = Path(results_folder)
    if not base_results_dir.is_absolute():
        base_results_dir = (Path.cwd() / base_results_dir).resolve()
    else:
        base_results_dir = base_results_dir.resolve()

    aggregated_dir = base_results_dir / "_aggregated"
    bootstrap_dir = aggregated_dir / "bootstrap_ranking"
    return base_results_dir, aggregated_dir, bootstrap_dir


def load_relative_error_dict_legacy(
    aggregated_dir: Path,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    rel_path = aggregated_dir / "relative_error_dict.pkl"
    if not rel_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {rel_path}")

    with open(rel_path, "rb") as f_id:
        relative_error_dict_raw = pickle.load(f_id)

    # Keep the legacy format used in figures.py:
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

            # Mirror figures.py threshold logic.
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


def compute_bootstrap_rank_positions(
    bootstrap_ranks_df: pd.DataFrame,
    ordered_strategies: List[str],
) -> pd.DataFrame:
    order_lookup = {strategy: idx for idx, strategy in enumerate(ordered_strategies)}
    position_rows = []

    for _, row in bootstrap_ranks_df[ordered_strategies].iterrows():
        valid_entries = [
            (strategy, float(row[strategy]))
            for strategy in ordered_strategies
            if pd.notna(row[strategy])
        ]
        valid_entries.sort(key=lambda item: (item[1], order_lookup[item[0]]))

        position_row = {strategy: np.nan for strategy in ordered_strategies}
        for rank_position, (strategy, _) in enumerate(valid_entries, start=1):
            position_row[strategy] = rank_position
        position_rows.append(position_row)

    return pd.DataFrame(position_rows, columns=ordered_strategies)


def compute_pairwise_relation_matrix(
    probability_matrix: pd.DataFrame,
    strategies: List[str],
    dominance_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relation_matrix = pd.DataFrame("", index=strategies, columns=strategies, dtype=object)
    strong_dominance = pd.DataFrame(False, index=strategies, columns=strategies, dtype=bool)

    for strat_a in strategies:
        for strat_b in strategies:
            if strat_a == strat_b:
                relation_matrix.loc[strat_a, strat_b] = "="
                continue

            p_ab = probability_matrix.loc[strat_a, strat_b]
            p_ba = probability_matrix.loc[strat_b, strat_a]
            if not np.isfinite(p_ab):
                relation_matrix.loc[strat_a, strat_b] = ""
            elif p_ab >= dominance_threshold:
                relation_matrix.loc[strat_a, strat_b] = "D"
                strong_dominance.loc[strat_a, strat_b] = True
            elif np.isfinite(p_ba) and p_ba >= dominance_threshold:
                relation_matrix.loc[strat_a, strat_b] = "L"
            else:
                relation_matrix.loc[strat_a, strat_b] = "?"

    return relation_matrix, strong_dominance


def _dfs_finish(
    node: str,
    adjacency: Dict[str, List[str]],
    visited: Set[str],
    order: List[str],
) -> None:
    visited.add(node)
    for neighbor in adjacency[node]:
        if neighbor not in visited:
            _dfs_finish(neighbor, adjacency, visited, order)
    order.append(node)


def _dfs_collect(
    node: str,
    adjacency: Dict[str, List[str]],
    visited: Set[str],
    component: List[str],
) -> None:
    visited.add(node)
    component.append(node)
    for neighbor in adjacency[node]:
        if neighbor not in visited:
            _dfs_collect(neighbor, adjacency, visited, component)


def strongly_connected_components(
    nodes: List[str],
    strong_dominance: pd.DataFrame,
) -> List[List[str]]:
    adjacency = {
        node: [other for other in nodes if strong_dominance.loc[node, other]]
        for node in nodes
    }
    transpose = {
        node: [other for other in nodes if strong_dominance.loc[other, node]]
        for node in nodes
    }

    visited = set()
    order: List[str] = []
    for node in nodes:
        if node not in visited:
            _dfs_finish(node, adjacency, visited, order)

    visited = set()
    components: List[List[str]] = []
    for node in reversed(order):
        if node in visited:
            continue
        component: List[str] = []
        _dfs_collect(node, transpose, visited, component)
        components.append(component)

    return components


def compute_dominance_tiers(
    strong_dominance: pd.DataFrame,
    ordered_strategies: List[str],
    rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    rank_lookup = (
        rank_summary.set_index("strategy")["mean_global_rank"].astype(float).to_dict()
    )
    order_lookup = {strategy: idx for idx, strategy in enumerate(ordered_strategies)}
    remaining = list(ordered_strategies)
    tier_rows = []
    tier_idx = 1

    while remaining:
        components = strongly_connected_components(remaining, strong_dominance)
        source_components = []

        for component in components:
            comp_set = set(component)
            has_incoming = any(
                strong_dominance.loc[other, member]
                for other in remaining
                if other not in comp_set
                for member in component
            )
            if not has_incoming:
                source_components.append(component)

        tier_nodes = sorted(
            [node for component in source_components for node in component],
            key=lambda strategy: order_lookup[strategy],
        )
        if not tier_nodes:
            raise RuntimeError("Failed to extract dominance tier from pairwise matrix.")

        for component in source_components:
            component_size = len(component)
            for strategy in sorted(component, key=lambda name: order_lookup[name]):
                n_strong_wins = int(
                    sum(
                        bool(strong_dominance.loc[strategy, other])
                        for other in ordered_strategies
                        if other != strategy
                    )
                )
                n_strong_losses = int(
                    sum(
                        bool(strong_dominance.loc[other, strategy])
                        for other in ordered_strategies
                        if other != strategy
                    )
                )
                n_inconclusive = int(
                    max(0, len(ordered_strategies) - 1 - n_strong_wins - n_strong_losses)
                )
                tier_rows.append(
                    {
                        "tier": tier_idx,
                        "strategy": strategy,
                        "strategy_label": strategy_label(strategy),
                        "mean_global_rank": float(rank_lookup.get(strategy, np.nan)),
                        "n_strong_wins": n_strong_wins,
                        "n_strong_losses": n_strong_losses,
                        "n_inconclusive": n_inconclusive,
                        "source_component_size": component_size,
                    }
                )

        remaining = [strategy for strategy in remaining if strategy not in set(tier_nodes)]
        tier_idx += 1

    tier_df = pd.DataFrame(tier_rows)
    if tier_df.empty:
        return tier_df

    tier_df = tier_df.sort_values(
        by=["tier", "mean_global_rank", "strategy_label"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return tier_df


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


def _format_relation_matrix(relation_matrix: pd.DataFrame) -> str:
    printable = relation_matrix.copy()
    printable.index = [strategy_label(s) for s in printable.index]
    printable.columns = [strategy_label(s) for s in printable.columns]
    return printable.to_string()


def _format_dominance_tier_table(tier_df: pd.DataFrame) -> str:
    printable = tier_df.copy()
    printable = printable.rename(
        columns={
            "tier": "Tier",
            "strategy_label": "Strategy",
            "mean_global_rank": "Mean Rank",
            "n_strong_wins": "Strong Wins",
            "n_strong_losses": "Strong Losses",
            "n_inconclusive": "Inconclusive",
            "source_component_size": "Source SCC Size",
        }
    )[
        [
            "Tier",
            "Strategy",
            "Mean Rank",
            "Strong Wins",
            "Strong Losses",
            "Inconclusive",
            "Source SCC Size",
        ]
    ]
    return printable.to_string(
        index=False,
        formatters={
            "Tier": "{:d}".format,
            "Mean Rank": "{:.3f}".format,
            "Strong Wins": "{:d}".format,
            "Strong Losses": "{:d}".format,
            "Inconclusive": "{:d}".format,
            "Source SCC Size": "{:d}".format,
        },
    )


def save_readable_summary_txt(
    output_path: Path,
    rank_summary: pd.DataFrame,
    probability_sorted: pd.DataFrame,
    relation_sorted: pd.DataFrame,
    tier_df: pd.DataFrame,
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
    lines.append(f"Dominance threshold  : {args.dominance_threshold}")
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
    lines.append(
        "Pairwise dominance status matrix "
        f"(D: row strongly dominates, L: row strongly loses, ?: inconclusive at {args.dominance_threshold:.2f})"
    )
    lines.append("-" * 80)
    lines.append(_format_relation_matrix(relation_sorted))
    lines.append("")
    lines.append("Dominance tiers")
    lines.append("-" * 80)
    lines.append(_format_dominance_tier_table(tier_df))
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


def plot_dominance_stacked_bar(
    tier_df: pd.DataFrame,
    output_path: Path,
    dominance_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    if tier_df.empty:
        return

    plot_df = tier_df.sort_values(
        by=["tier", "mean_global_rank", "strategy_label"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    labels = plot_df["strategy_label"].tolist()
    strong_wins = plot_df["n_strong_wins"].astype(int).to_numpy()
    inconclusive = plot_df["n_inconclusive"].astype(int).to_numpy()
    strong_losses = plot_df["n_strong_losses"].astype(int).to_numpy()

    cm = 1 / 2.54
    font_size = 8
    fig, ax = plt.subplots(figsize=(8.75 * cm, 7.5 * cm))
    tier_gap = 0.55
    bar_height = 0.72
    y = []
    current_y = 0.0
    previous_tier = None
    for tier in plot_df["tier"].tolist():
        if previous_tier is not None and tier != previous_tier:
            current_y += tier_gap
        y.append(current_y)
        current_y += 1.0
        previous_tier = tier
    y = np.asarray(y, dtype=float)

    ax.barh(
        y,
        strong_wins,
        color="#2a9d8f",
        edgecolor="white",
        linewidth=0.4,
        height=bar_height,
        label="Strong wins",
    )
    ax.barh(
        y,
        inconclusive,
        left=strong_wins,
        color="#d9d9d9",
        edgecolor="white",
        linewidth=0.4,
        height=bar_height,
        label="Inconclusive",
    )
    ax.barh(
        y,
        strong_losses,
        left=strong_wins + inconclusive,
        color="#d55e5e",
        edgecolor="white",
        linewidth=0.4,
        height=bar_height,
        label="Strong losses",
    )

    x_right = int(
        plot_df[["n_strong_wins", "n_inconclusive", "n_strong_losses"]]
        .sum(axis=1)
        .max()
    )
    tier_groups = plot_df.groupby("tier", sort=True)
    for tier, group in tier_groups:
        group_positions = y[group.index.to_numpy()]
        ymin = float(group_positions.min() - 0.5)
        ymax = float(group_positions.max() + 0.5)
        ax.axhspan(ymin, ymax, color="black", alpha=0.035, lw=0, zorder=0)
        ax.axhline(ymin, color="#7f7f7f", linewidth=0.45, alpha=0.7, zorder=1)
        ax.axhline(ymax, color="#7f7f7f", linewidth=0.45, alpha=0.7, zorder=1)
        ax.text(
            x_right + 0.35,
            0.5 * (ymin + ymax),
            f"Tier {tier}",
            va="center",
            ha="left",
            fontsize=font_size,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=font_size)
    ax.invert_yaxis()
    ax.set_xlim(0, x_right + 1.6)
    ax.set_xticks(np.arange(0, x_right + 1, 2))
    ax.tick_params(width=0.3, which="both", labelsize=font_size)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.grid(axis="x", linestyle="--", linewidth=0.2, alpha=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#2a9d8f", ec="white", lw=0.4, label="Strong wins"),
        plt.Rectangle((0, 0), 1, 1, fc="#d9d9d9", ec="white", lw=0.4, label="Inconclusive"),
        plt.Rectangle((0, 0), 1, 1, fc="#d55e5e", ec="white", lw=0.4, label="Strong losses"),
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        bbox_to_anchor=(0.55, 0.015),
        loc="lower center",
        fontsize=font_size,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
        handletextpad=0.4,
    )
    fig.text(0.55, 0.14, "Number of pairwise comparisons", ha="center", fontsize=font_size)

    plt.subplots_adjust(
        left=0.31,
        right=0.93,
        top=0.95,
        bottom=0.24,
    )
    fig.savefig(output_path)
    plt.close(fig)


def load_saved_rank_summary(bootstrap_dir: Path) -> pd.DataFrame:
    path = bootstrap_dir / RANK_SUMMARY_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved rank summary: {path}. Run the full bootstrap flow first."
        )
    return pd.read_csv(path)


def load_saved_probability_matrix(bootstrap_dir: Path) -> pd.DataFrame:
    path = bootstrap_dir / PAIRWISE_PROBABILITY_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing saved pairwise probability matrix: {path}. "
            "Run the full bootstrap flow first."
        )
    return pd.read_csv(path, index_col=0)


def save_bootstrap_metadata(bootstrap_dir: Path, payload: Dict[str, object]) -> None:
    path = bootstrap_dir / BOOTSTRAP_METADATA_NAME
    with open(path, "w", encoding="utf-8") as f_id:
        json.dump(payload, f_id, indent=2)


def load_bootstrap_metadata(bootstrap_dir: Path) -> Dict[str, object]:
    path = bootstrap_dir / BOOTSTRAP_METADATA_NAME
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f_id:
        return json.load(f_id)


def remove_legacy_outputs(output_dir: Path) -> None:
    for name in LEGACY_OUTPUTS_TO_REMOVE:
        path = output_dir / name
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()
    base_results_dir, aggregated_dir, bootstrap_dir = resolve_results_dirs(
        args.results_folder
    )
    summary_args = argparse.Namespace(**vars(args))
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_existing_probability:
        remove_legacy_outputs(bootstrap_dir)

    print("[start] bootstrap ranking analysis")
    print(f"  results dir          : {base_results_dir}")
    print(f"  aggregated dir       : {aggregated_dir}")
    print(f"  output dir           : {bootstrap_dir}")
    print(f"  reuse probability    : {args.reuse_existing_probability}")
    print(f"  n_bootstraps         : {args.n_bootstraps}")
    print(f"  random seed          : {args.seed}")
    print(f"  captured_ls          : {args.captured_ls}")
    print(f"  threshold_factor     : {args.threshold_factor}")
    print(f"  required_consecutive : {args.required_consecutive}")
    print(f"  n_seeds_per_strategy : {args.n_seeds_per_strategy}")
    print(f"  dominance_threshold  : {args.dominance_threshold}")
    threshold_dict = {}
    bootstrap_ranks_df = None
    bootstrap_rank_positions_df = None

    if args.reuse_existing_probability:
        metadata = load_bootstrap_metadata(bootstrap_dir)
        if metadata:
            for key in [
                "n_bootstraps",
                "seed",
                "captured_ls",
                "threshold_factor",
                "required_consecutive",
                "n_seeds_per_strategy",
            ]:
                if key in metadata:
                    setattr(summary_args, key, metadata[key])
        rank_summary = load_saved_rank_summary(bootstrap_dir)
        if "mean_global_rank" not in rank_summary.columns or "strategy" not in rank_summary.columns:
            raise RuntimeError(
                f"{bootstrap_dir / RANK_SUMMARY_CSV_NAME} does not contain the required columns."
            )
        rank_summary = rank_summary.sort_values(
            "mean_global_rank", ascending=True, kind="mergesort"
        ).reset_index(drop=True)
        ordered_strategies = rank_summary["strategy"].tolist()
        probability_matrix = load_saved_probability_matrix(bootstrap_dir)
        missing = [
            strategy for strategy in ordered_strategies if strategy not in probability_matrix.index
        ]
        if missing:
            raise RuntimeError(
                "Saved pairwise probability matrix is missing strategies: "
                f"{missing}"
            )
        probability_matrix = probability_matrix.loc[ordered_strategies, ordered_strategies]
        print(f"  strategies           : {ordered_strategies}")
        print("  excluded threshold   : [reused existing probability matrix]")
        if metadata:
            print(f"  source n_bootstraps  : {metadata.get('n_bootstraps')}")
    else:
        relative_error_dict = load_relative_error_dict_legacy(aggregated_dir)
        available_cases = [c for c in CASE_STUDIES if c in relative_error_dict]
        if not available_cases:
            raise RuntimeError(
                "No case data found in relative_error_dict.pkl. "
                "Run output_files.py first."
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
        ).sort_values("mean_global_rank", ascending=True, kind="mergesort")
        ordered_strategies = rank_summary["strategy"].tolist()
        bootstrap_rank_positions_df = compute_bootstrap_rank_positions(
            bootstrap_ranks_df=bootstrap_ranks_df,
            ordered_strategies=ordered_strategies,
        )
        bootstrap_rank_positions_df.insert(0, "bootstrap_id", np.arange(1, len(bootstrap_rank_positions_df) + 1))
        rank_position_summary = pd.DataFrame(
            {
                "strategy": ordered_strategies,
                "mean_rank_position": [
                    float(np.nanmean(bootstrap_rank_positions_df[s])) for s in ordered_strategies
                ],
                "median_rank_position": [
                    float(np.nanmedian(bootstrap_rank_positions_df[s])) for s in ordered_strategies
                ],
                "p2_5_rank_position": [
                    float(np.nanpercentile(bootstrap_rank_positions_df[s], 2.5)) for s in ordered_strategies
                ],
                "p97_5_rank_position": [
                    float(np.nanpercentile(bootstrap_rank_positions_df[s], 97.5)) for s in ordered_strategies
                ],
                "n_valid_bootstraps": [
                    int(bootstrap_rank_positions_df[s].notna().sum()) for s in ordered_strategies
                ],
            }
        ).sort_values("mean_rank_position", ascending=True, kind="mergesort")
        probability_matrix = compute_pairwise_probability_matrix(
            bootstrap_ranks_df=bootstrap_ranks_df,
            strategies=strategies,
        )
        rank_summary.to_csv(bootstrap_dir / RANK_SUMMARY_CSV_NAME, index=False)
        bootstrap_rank_positions_df.to_csv(bootstrap_dir / RANK_POSITIONS_CSV_NAME, index=False)
        rank_position_summary.to_csv(bootstrap_dir / RANK_POSITION_SUMMARY_CSV_NAME, index=False)
        save_bootstrap_metadata(
            bootstrap_dir,
            {
                "n_bootstraps": args.n_bootstraps,
                "seed": args.seed,
                "captured_ls": args.captured_ls,
                "threshold_factor": args.threshold_factor,
                "required_consecutive": args.required_consecutive,
                "n_seeds_per_strategy": args.n_seeds_per_strategy,
                "include_high_dimensional": bool(args.include_high_dimensional),
                "cases": list(cases_for_bootstrap),
                "strategies": list(strategies),
            },
        )

    relation_matrix, strong_dominance = compute_pairwise_relation_matrix(
        probability_matrix=probability_matrix,
        strategies=rank_summary["strategy"].tolist(),
        dominance_threshold=args.dominance_threshold,
    )

    ordered_strategies = rank_summary["strategy"].tolist()
    probability_sorted = probability_matrix.loc[ordered_strategies, ordered_strategies]
    relation_sorted = relation_matrix.loc[ordered_strategies, ordered_strategies]
    tier_df = compute_dominance_tiers(
        strong_dominance=strong_dominance.loc[ordered_strategies, ordered_strategies],
        ordered_strategies=ordered_strategies,
        rank_summary=rank_summary,
    )

    print("")
    print("--- Pairwise Probability Matrix: P(Row Strategy is Better than Column Strategy) ---")
    probability_print = probability_sorted.round(3).copy()
    probability_print.index = [strategy_label(s) for s in probability_print.index]
    probability_print.columns = [strategy_label(s) for s in probability_print.columns]
    print(probability_print.to_string())
    print("")
    print(
        "--- Pairwise Dominance Status "
        f"(D/L/? at threshold {args.dominance_threshold:.2f}) ---"
    )
    print(_format_relation_matrix(relation_sorted))
    print("")
    print("--- Dominance Tiers ---")
    print(_format_dominance_tier_table(tier_df))

    probability_sorted.to_csv(bootstrap_dir / PAIRWISE_PROBABILITY_CSV_NAME, float_format="%.8f")
    relation_sorted.to_csv(bootstrap_dir / PAIRWISE_RELATION_CSV_NAME)
    tier_df.to_csv(bootstrap_dir / DOMINANCE_TIERS_CSV_NAME, index=False)
    plot_dominance_stacked_bar(
        tier_df=tier_df,
        output_path=bootstrap_dir / DOMINANCE_BAR_PDF_NAME,
        dominance_threshold=args.dominance_threshold,
    )

    summary_txt_path = bootstrap_dir / SUMMARY_TXT_NAME
    save_readable_summary_txt(
        output_path=summary_txt_path,
        rank_summary=rank_summary,
        probability_sorted=probability_sorted,
        relation_sorted=relation_sorted,
        tier_df=tier_df,
        threshold_dict=threshold_dict,
        args=summary_args,
    )

    if not args.no_plots:
        if bootstrap_ranks_df is not None:
            plot_bootstrap_rank_boxplot(
                bootstrap_ranks_df=bootstrap_ranks_df,
                ordered_strategies=ordered_strategies,
                output_path=bootstrap_dir / "bootstrap_rank_boxplot.pdf",
                n_bootstraps=args.n_bootstraps,
            )
        else:
            print("  [skip] bootstrap_rank_boxplot.pdf requires full bootstrap recomputation.")

    print("")
    print("[done] bootstrap artifacts written:")
    print(f"  - {summary_txt_path}")
    print(f"  - {bootstrap_dir / RANK_SUMMARY_CSV_NAME}")
    if bootstrap_rank_positions_df is not None:
        print(f"  - {bootstrap_dir / RANK_POSITIONS_CSV_NAME}")
        print(f"  - {bootstrap_dir / RANK_POSITION_SUMMARY_CSV_NAME}")
    print(f"  - {bootstrap_dir / PAIRWISE_PROBABILITY_CSV_NAME}")
    print(f"  - {bootstrap_dir / PAIRWISE_RELATION_CSV_NAME}")
    print(f"  - {bootstrap_dir / DOMINANCE_TIERS_CSV_NAME}")
    print(f"  - {bootstrap_dir / DOMINANCE_BAR_PDF_NAME}")
    print(f"  - {bootstrap_dir / BOOTSTRAP_METADATA_NAME}")
    if not args.no_plots:
        if bootstrap_ranks_df is not None:
            print(f"  - {bootstrap_dir / 'bootstrap_rank_boxplot.pdf'}")


if __name__ == "__main__":
    main()
