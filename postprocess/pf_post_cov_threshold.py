import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data import resolve_cpu_workers
from active_learning.eier import build_gp_cache_from_gpr, estimate_pf_posterior_samples
from limit_states import REGISTRY as ls_REGISTRY
from reconstruct_gp_step import (
    build_pf_model_report,
    derive_stream_seed,
    estimate_pf_model_stream,
    fit_target_model,
    get_saved_metric,
    load_training_samples,
    try_load_exact_checkpoint,
)
from settings import (
    BASE_RESULTS_DIR,
    CAPTURED_LS,
    CASE_STUDIES,
    CASE_TITLES,
    DEFAULT_POST_N_G_PF,
    DEFAULT_POST_N_PF_POST_POOL,
    DEFAULT_POST_PF_POST_BATCH_SIZE,
    DEFAULT_POST_PREDICT_BATCH_SIZE,
    DEFAULT_POST_PREDICT_N_JOBS,
    DEFAULT_STRATEGIES,
    DOE_SAMPLES,
    EIER_REFERENCE_STRATEGY,
    GROUP_2D,
    PF_POST_COV_SUMMARY_NAME,
    PF_POST_SAMPLES_DIRNAME,
    PF_POST_COV_TABLE_NAME,
    REAL_PF_VALUES,
    REQUIRED_CONSECUTIVE,
    REPO_ROOT,
    STRATEGY_RANKINGS_TABLE_NAME,
    THRESHOLD_DICT_NAME,
    THRESHOLD_FACTOR,
    THRESHOLD_HITS_TABLE_NAME,
    strategy_label,
    strategy_to_dir,
)


try:
    DEFAULT_RESULTS_FOLDER = str(BASE_RESULTS_DIR.relative_to(REPO_ROOT))
except ValueError:
    DEFAULT_RESULTS_FOLDER = str(BASE_RESULTS_DIR)

REPORT_LINES = []


def progress(message=""):
    print(str(message))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pf_post_CoV per seed at the saved threshold-hit train size when available, "
            "or at the final train size otherwise. Threshold-hit metadata is reused from the "
            "aggregated postprocess artifacts, while GP-based posterior metrics can be recomputed "
            "with a streamed estimator and checkpointed incrementally."
        )
    )
    parser.add_argument(
        "--results-folder",
        "--results_folder",
        dest="results_folder",
        default=DEFAULT_RESULTS_FOLDER,
        help=(
            "Folder under the repository root containing case-study results "
            f"(default: {DEFAULT_RESULTS_FOLDER})."
        ),
    )
    parser.add_argument(
        "--case-study",
        dest="case_studies_filter",
        action="append",
        choices=CASE_STUDIES,
        help="Optional case-study filter. Repeat the flag to select multiple cases.",
    )
    parser.add_argument(
        "--strategy",
        dest="strategies_filter",
        action="append",
        choices=DEFAULT_STRATEGIES,
        help="Optional strategy filter. Repeat the flag to select multiple strategies.",
    )
    parser.add_argument(
        "--max-runs-per-strategy",
        type=int,
        default=None,
        help=(
            "Optional limit on the number of sorted run folders evaluated per case/strategy. "
            "Thresholds still come from the full aggregated campaign when available."
        ),
    )
    parser.add_argument(
        "--n-g-pf",
        type=int,
        default=None,
        help="Override the number of posterior Pf trajectories to sample when reconstruction is needed.",
    )
    parser.add_argument(
        "--n-pf-post-pool",
        type=int,
        default=None,
        help="Override the fixed support size used for posterior Pf estimation when reconstruction is needed.",
    )
    parser.add_argument(
        "--pf-post-batch-size",
        type=int,
        default=None,
        help="Override the posterior Pf support chunk size used when reconstruction is needed.",
    )
    parser.add_argument(
        "--posterior-workers",
        type=int,
        default=None,
        help="Override the number of worker threads used by the streamed posterior Pf estimator.",
    )
    parser.add_argument(
        "--posterior-estimator",
        choices=["streamed", "dense"],
        default="streamed",
        help="Posterior Pf estimator to use whenever Pf_post_* must be recomputed.",
    )
    parser.add_argument(
        "--force-recompute-posterior",
        action="store_true",
        help=(
            "Ignore saved Pf_post_* values in output.json and recompute them so all runs use the same "
            "posterior estimator/settings."
        ),
    )
    parser.add_argument(
        "--save-pf-post-samples",
        action="store_true",
        help=(
            "For recomputed posterior rows, save the full posterior Pf sample vector as a compressed "
            "artifact per evaluated run."
        ),
    )
    parser.add_argument(
        "--verify-pf-model",
        action="store_true",
        help=(
            "For recomputed posterior rows only, also recompute Pf_model with the saved config n_mcs_pf "
            "and compare it against output.json."
        ),
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=None,
        help="Override predict_batch_size used for Pf-model verification batches.",
    )
    parser.add_argument(
        "--predict-n-jobs",
        type=int,
        default=None,
        help="Override predict_n_jobs used for Pf-model verification.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any saved progress file and start the campaign from scratch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional per-run details beyond the compact campaign progress line.",
    )
    return parser.parse_args()


def report(message=""):
    text = str(message)
    print(text)
    REPORT_LINES.append(text)


def flush_report_summary(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_id:
        f_id.write("\n".join(REPORT_LINES).rstrip() + "\n")
    print(f"[save][SUMMARY] {path}")


def progress_path_for(table_path: Path):
    return table_path.with_suffix(".progress.jsonl")


def metadata_path_for(table_path: Path):
    return table_path.with_name(f"{table_path.stem}.metadata.json")


def remove_if_exists(path: Path):
    if path.is_file():
        path.unlink()


def campaign_settings_dict(args, base_results_dir):
    return {
        "base_results_dir": str(base_results_dir),
        "case_studies_filter": args.case_studies_filter,
        "strategies_filter": args.strategies_filter,
        "max_runs_per_strategy": args.max_runs_per_strategy,
        "n_g_pf_override": args.n_g_pf,
        "n_pf_post_pool_override": args.n_pf_post_pool,
        "pf_post_batch_size_override": args.pf_post_batch_size,
        "posterior_workers_override": args.posterior_workers,
        "posterior_estimator": args.posterior_estimator,
        "force_recompute_posterior": bool(args.force_recompute_posterior),
        "save_pf_post_samples": bool(args.save_pf_post_samples),
        "verify_pf_model": bool(args.verify_pf_model),
        "predict_batch_size_override": args.predict_batch_size,
        "predict_n_jobs_override": args.predict_n_jobs,
    }


def write_metadata(path: Path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_id:
        json.dump(metadata, f_id, indent=2)


def load_metadata(path: Path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f_id:
        return json.load(f_id)


def settings_compatible(saved_settings, current_settings):
    if saved_settings is None:
        return True
    scope_only_keys = {
        "case_studies_filter",
        "strategies_filter",
        "max_runs_per_strategy",
        "posterior_workers_override",
    }
    saved_core = {k: v for k, v in saved_settings.items() if k not in scope_only_keys}
    current_core = {k: v for k, v in current_settings.items() if k not in scope_only_keys}
    return json.dumps(saved_core, sort_keys=True) == json.dumps(current_core, sort_keys=True)


def append_progress_row(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f_id:
        f_id.write(json.dumps(row, sort_keys=True) + "\n")


def load_progress_rows(path: Path):
    if not path.is_file():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f_id:
        for line in f_id:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_row_key(case, strategy, run_name, evaluation_train_size):
    return f"{case}::{strategy}::{run_name}::{int(evaluation_train_size)}"


def pf_post_samples_root_for(aggregated_dir: Path):
    return aggregated_dir / PF_POST_SAMPLES_DIRNAME


def pf_post_samples_path_for(
    samples_root: Path,
    case: str,
    strategy: str,
    run_name: str,
    evaluation_step: int,
    evaluation_train_size: int,
    posterior_estimator: str,
    n_g_pf: int,
    n_pf_post_pool: int,
):
    filename = (
        f"{run_name}__step{int(evaluation_step):04d}__train{int(evaluation_train_size):04d}"
        f"__{posterior_estimator}__ng{int(n_g_pf)}__pool{int(n_pf_post_pool)}.npz"
    )
    return samples_root / case / strategy / filename


def save_pf_post_samples(path: Path, pf_samples: np.ndarray, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pf_samples": np.asarray(pf_samples, dtype=np.float64)}
    payload.update(metadata)
    np.savez_compressed(path, **payload)


def row_key_from_row(row):
    return build_row_key(row["case"], row["strategy"], row["run"], row["evaluation_train_size"])


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda r: (
            CASE_STUDIES.index(r["case"]) if r["case"] in CASE_STUDIES else 10**9,
            DEFAULT_STRATEGIES.index(r["strategy"]) if r["strategy"] in DEFAULT_STRATEGIES else 10**9,
            r["exp_num"],
            r["run"],
        ),
    )


def update_counts_from_row(row, source_counts, eval_mode_counts, evaluation_source_counts):
    source_counts[row["posterior_source"]] = source_counts.get(row["posterior_source"], 0) + 1
    eval_mode_counts[row["evaluation_mode"]] = eval_mode_counts.get(row["evaluation_mode"], 0) + 1
    evaluation_source_counts[row["evaluation_source"]] = evaluation_source_counts.get(row["evaluation_source"], 0) + 1


def plan_evaluations(run_records, threshold_dict, threshold_hit_lookup):
    planned_jobs = []
    skipped_reason_counts = {}

    for case, strategy_dict in run_records.items():
        if case not in threshold_dict:
            skipped_reason_counts["missing_threshold_for_case"] = skipped_reason_counts.get("missing_threshold_for_case", 0) + sum(
                len(runs) for runs in strategy_dict.values()
            )
            continue

        threshold_info = threshold_dict[case]
        threshold = float(threshold_info["threshold_delta_pf"])
        for strategy, runs in strategy_dict.items():
            for run_name, record in sorted(runs.items()):
                evaluation_info = threshold_hit_lookup.get((case, strategy, run_name))
                if evaluation_info is None:
                    evaluation_info, skip_reason = derive_evaluation_from_output(
                        case=case,
                        config=record["config"],
                        output=record["output"],
                        threshold=threshold,
                    )
                    if evaluation_info is None:
                        skipped_reason_counts[skip_reason] = skipped_reason_counts.get(skip_reason, 0) + 1
                        continue

                planned_jobs.append(
                    {
                        "case": case,
                        "strategy": strategy,
                        "run_name": run_name,
                        "record": record,
                        "threshold_info": threshold_info,
                        "evaluation_info": evaluation_info,
                    }
                )

    return planned_jobs, skipped_reason_counts


def resolve_resume_state(progress_path: Path, metadata_path: Path, current_settings, overwrite: bool):
    if overwrite:
        remove_if_exists(progress_path)
        remove_if_exists(metadata_path)
        return [], set(), None

    progress_rows = load_progress_rows(progress_path)
    saved_metadata = load_metadata(metadata_path)
    saved_settings = None if saved_metadata is None else saved_metadata.get("campaign_settings")

    if progress_rows and not settings_compatible(saved_settings, current_settings):
        raise ValueError(
            "Existing progress file was created with different campaign settings. "
            "Use --overwrite to start a new run or restore the previous settings."
        )

    completed_keys = {row_key_from_row(row) for row in progress_rows}
    return progress_rows, completed_keys, saved_metadata


def resolve_results_dirs(results_folder):
    base_results_dir = Path(results_folder)
    if not base_results_dir.is_absolute():
        base_results_dir = REPO_ROOT / base_results_dir
    base_results_dir = base_results_dir.resolve()
    aggregated_dir = base_results_dir / "_aggregated"
    return base_results_dir, aggregated_dir


def parse_experiment_number(run_key):
    text = str(run_key)
    tokens = text.split("_")

    for idx in range(len(tokens) - 1):
        if tokens[idx].isdigit() and tokens[idx + 1].isdigit():
            return int(tokens[idx + 1])

    for token in tokens:
        if token.isdigit():
            return int(token)

    return text



def case_budget(case):
    return 200 if case in GROUP_2D else 500



def fmt_sci_compact(x):
    mantissa, exponent = f"{float(x):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"



def load_run_records(
    base_results_dir,
    max_runs_per_strategy=None,
    emit_progress=True,
    case_studies_filter=None,
    strategies_filter=None,
):
    run_records = {}
    missing_files = []
    runs_seen = 0
    runs_loaded = 0

    selected_cases = CASE_STUDIES if not case_studies_filter else [case for case in CASE_STUDIES if case in case_studies_filter]
    selected_strategies = DEFAULT_STRATEGIES if not strategies_filter else [s for s in DEFAULT_STRATEGIES if s in strategies_filter]

    for case in selected_cases:
        case_dir = base_results_dir / case
        if not case_dir.is_dir():
            if emit_progress:
                report(f"[skip][case] missing directory: {case_dir}")
            continue

        case_records = {}
        if emit_progress:
            report(f"[case] {case}")
        for strategy in selected_strategies:
            method_dir_name = strategy_to_dir(strategy)
            method_dir_candidates = [case_dir / method_dir_name]
            if method_dir_name != strategy:
                method_dir_candidates.append(case_dir / strategy)

            method_dir = None
            for candidate in method_dir_candidates:
                if candidate.is_dir():
                    method_dir = candidate
                    break

            if method_dir is None:
                if emit_progress:
                    report(f"  [skip][strategy] {strategy} -> no directory")
                continue

            run_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
            if max_runs_per_strategy is not None:
                run_dirs = run_dirs[: max(0, int(max_runs_per_strategy))]

            if not run_dirs:
                if emit_progress:
                    report(f"  [skip][strategy] {strategy} -> no runs")
                continue

            strategy_records = {}
            if emit_progress:
                report(f"  [strategy] {strategy} -> {method_dir.name} | runs={len(run_dirs)}")
            for run_dir in run_dirs:
                runs_seen += 1
                run_name = run_dir.name
                config_path = run_dir / "config.json"
                output_path = run_dir / "output.json"

                if not config_path.is_file():
                    missing_files.append(
                        {
                            "case": case,
                            "strategy": strategy,
                            "run": run_name,
                            "missing": "config.json",
                        }
                    )
                    continue
                if not output_path.is_file():
                    missing_files.append(
                        {
                            "case": case,
                            "strategy": strategy,
                            "run": run_name,
                            "missing": "output.json",
                        }
                    )
                    continue

                with open(config_path, "r", encoding="utf-8") as f_id:
                    config = json_load(f_id)
                with open(output_path, "r", encoding="utf-8") as f_id:
                    output = json_load(f_id)

                strategy_records[run_name] = {
                    "run_dir": run_dir,
                    "config": config,
                    "output": output,
                }
                runs_loaded += 1

            if strategy_records:
                case_records[strategy] = strategy_records
        if case_records:
            run_records[case] = case_records
        if emit_progress:
            report("")

    return run_records, missing_files, runs_seen, runs_loaded


def json_load(file_obj):
    return json.load(file_obj)



def parse_optional_int(value):
    if value in (None, "", "None", "nan", "NaN", "<NA>"):
        return None
    return int(float(value))



def parse_optional_float(value):
    if value in (None, "", "None", "nan", "NaN", "<NA>"):
        return None
    return float(value)



def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, "", "None", "nan", "NaN", "<NA>"):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")



def normalize_relative_error_dict(relative_error_dict_raw):
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



def build_relative_error_dict(run_records):
    relative_error_dict = {}

    for case, strategy_dict in run_records.items():
        pf_ref = REAL_PF_VALUES.get(case)
        if pf_ref is None or pf_ref <= 0.0:
            continue

        case_dict = {}
        for strategy, runs in strategy_dict.items():
            run_dict = {}
            for run_name, record in runs.items():
                pf_model = np.asarray(record["output"].get("Pf_model", []), dtype=float)
                if pf_model.size == 0:
                    continue
                abs_rel_error = np.abs((pf_model - pf_ref) / pf_ref)
                run_dict[run_name] = abs_rel_error.tolist()
            if run_dict:
                case_dict[strategy] = run_dict
        if case_dict:
            relative_error_dict[case] = case_dict

    return relative_error_dict



def load_threshold_relative_error_dict(base_results_dir, aggregated_dir, eval_run_records, max_runs_per_strategy):
    rel_path = aggregated_dir / "relative_error_dict.pkl"
    if rel_path.is_file():
        with open(rel_path, "rb") as f_id:
            relative_error_dict_raw = pickle.load(f_id)
        return normalize_relative_error_dict(relative_error_dict_raw), str(rel_path), False

    report(f"[warn] missing aggregated relative-error artifact: {rel_path}")
    if max_runs_per_strategy is None:
        return build_relative_error_dict(eval_run_records), "evaluation run subset", True

    report("[thresholds] loading the full results-folder to recover full-dataset thresholds")
    threshold_run_records, _, _, _ = load_run_records(
        base_results_dir,
        max_runs_per_strategy=None,
        emit_progress=False,
    )
    return build_relative_error_dict(threshold_run_records), "full raw results-folder fallback", True



def normalize_threshold_dict(raw_threshold_dict):
    threshold_dict = {}
    for case, data in raw_threshold_dict.items():
        threshold_dict[case] = {
            "case": case,
            "case_title": data.get("case_title", CASE_TITLES.get(case, case)),
            "captured_ls": int(data.get("captured_ls", CAPTURED_LS)),
            "threshold_factor": float(data.get("threshold_factor", THRESHOLD_FACTOR)),
            "threshold_delta_pf": float(data["threshold_delta_pf"]),
            "required_consecutive": int(data.get("required_consecutive", REQUIRED_CONSECUTIVE)),
            "threshold_entry": dict(data.get("threshold_entry", {})),
            "topk_minima": [dict(entry) for entry in data.get("topk_minima", [])],
            "target_epsilon": list(data.get("target_epsilon", [])),
        }
    return threshold_dict



def load_threshold_json(threshold_json_path):
    if not threshold_json_path.is_file():
        return None
    with open(threshold_json_path, "r", encoding="utf-8") as f_id:
        raw_threshold_dict = json.load(f_id)
    return normalize_threshold_dict(raw_threshold_dict)



def load_threshold_hit_rows(threshold_hits_path):
    if not threshold_hits_path.is_file():
        return []

    rows = []
    with open(threshold_hits_path, "r", encoding="utf-8", newline="") as f_id:
        reader = csv.DictReader(f_id, delimiter="	")
        for raw in reader:
            rows.append(
                {
                    "case": raw.get("case"),
                    "case_title": raw.get("case_title"),
                    "strategy": raw.get("strategy"),
                    "strategy_label": raw.get("strategy_label"),
                    "run": raw.get("run"),
                    "exp_num": parse_optional_int(raw.get("exp_num")),
                    "seed": parse_optional_int(raw.get("seed")),
                    "passive_samples": parse_optional_int(raw.get("passive_samples")),
                    "al_batch": parse_optional_int(raw.get("al_batch")),
                    "reached_threshold": parse_bool(raw.get("reached_threshold")),
                    "first_hit_idx": parse_optional_int(raw.get("first_hit_idx")),
                    "first_hit_step": parse_optional_int(raw.get("first_hit_step")),
                    "first_hit_samples": parse_optional_int(raw.get("first_hit_samples")),
                    "evaluation_mode": raw.get("evaluation_mode") or None,
                    "evaluation_step": parse_optional_int(raw.get("evaluation_step")),
                    "evaluation_train_size": parse_optional_int(raw.get("evaluation_train_size")),
                    "final_step": parse_optional_int(raw.get("final_step")),
                    "final_train_size": parse_optional_int(raw.get("final_train_size")),
                    "delta_at_hit": parse_optional_float(raw.get("delta_at_hit")),
                    "best_delta_pf": parse_optional_float(raw.get("best_delta_pf")),
                    "best_idx": parse_optional_int(raw.get("best_idx")),
                    "best_samples": parse_optional_int(raw.get("best_samples")),
                    "threshold_delta_pf": parse_optional_float(raw.get("threshold_delta_pf")),
                    "required_consecutive": parse_optional_int(raw.get("required_consecutive")),
                    "global_rank": parse_optional_int(raw.get("global_rank")),
                    "evaluation_source": "saved_threshold_hits",
                }
            )
    return rows



def build_threshold_hit_lookup(rows):
    lookup = {}
    for row in rows:
        key = (row.get("case"), row.get("strategy"), row.get("run"))
        if None not in key:
            lookup[key] = row
    return lookup



def build_threshold_dict(relative_error_dict):
    threshold_dict = {}
    excluded_from_threshold = {EIER_REFERENCE_STRATEGY}

    for case in CASE_STUDIES:
        if case not in relative_error_dict:
            continue

        per_strategy_minima = []
        max_len_case = case_budget(case)

        for strategy, exp_dict in relative_error_dict[case].items():
            if strategy in excluded_from_threshold:
                continue
            if not exp_dict:
                continue

            max_len_available = max(len(arr) for arr in exp_dict.values())
            cap_len = min(max_len_available, max_len_case)
            if cap_len <= 0:
                continue

            rel_diff_mat = np.full((len(exp_dict), cap_len), np.nan, dtype=float)
            for row_idx, (_, rel_diff) in enumerate(sorted(exp_dict.items())):
                this_len = min(len(rel_diff), cap_len)
                rel_diff_mat[row_idx, :this_len] = rel_diff[:this_len]

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
            report(f"[warn] no valid threshold minima for case '{case}'.")
            continue

        per_strategy_minima_sorted = sorted(per_strategy_minima, key=lambda d: d["delta_pf_min"])
        top_k = min(CAPTURED_LS, len(per_strategy_minima_sorted))
        topk = []
        for rank in range(top_k):
            entry = per_strategy_minima_sorted[rank].copy()
            entry["rank"] = rank + 1
            topk.append(entry)

        threshold_rank_index = min(CAPTURED_LS - 1, top_k - 1)
        threshold_entry = topk[threshold_rank_index]
        threshold_delta_pf = THRESHOLD_FACTOR * float(threshold_entry["delta_pf_min"])

        threshold_dict[case] = {
            "topk_minima": topk,
            "captured_ls": CAPTURED_LS,
            "threshold_factor": THRESHOLD_FACTOR,
            "threshold_delta_pf": threshold_delta_pf,
            "threshold_entry": threshold_entry,
        }

    return threshold_dict



def find_first_hit_index(rel_diff, threshold, required_consecutive):
    rel_diff = np.asarray(rel_diff, dtype=float)
    if rel_diff.size == 0:
        return None

    is_below = rel_diff <= threshold
    max_start = rel_diff.size - required_consecutive
    if max_start < 0:
        return None

    for idx in range(max_start + 1):
        if np.all(is_below[idx : idx + required_consecutive]):
            return idx
    return None



def derive_evaluation_from_output(case, config, output, threshold):
    pf_ref = REAL_PF_VALUES.get(case)
    if pf_ref is None or pf_ref <= 0.0:
        return None, "missing_pf_reference"

    pf_model = np.asarray(output.get("Pf_model", []), dtype=float)
    if pf_model.size == 0:
        return None, "missing_pf_model"

    rel_diff = np.abs((pf_model - pf_ref) / pf_ref)
    cap_len = min(rel_diff.size, case_budget(case))
    if cap_len <= 0:
        return None, "empty_relative_error"
    rel_diff = rel_diff[:cap_len]

    passive_samples = int(config.get("passive_samples", DOE_SAMPLES))
    al_batch = int(config.get("al_batch", 1))
    first_hit_idx = find_first_hit_index(
        rel_diff,
        threshold=threshold,
        required_consecutive=REQUIRED_CONSECUTIVE,
    )
    final_step = int(cap_len - 1)
    final_train_size = int(passive_samples + final_step * al_batch)
    no_hit_value = 201 if case in GROUP_2D else 501

    if first_hit_idx is not None:
        reached_threshold = True
        first_hit_step = int(first_hit_idx + (REQUIRED_CONSECUTIVE - 1))
        first_hit_samples = int(passive_samples + first_hit_step * al_batch)
        evaluation_mode = "threshold_hit"
        evaluation_step = int(first_hit_step)
        evaluation_train_size = int(first_hit_samples)
    else:
        reached_threshold = False
        first_hit_step = None
        first_hit_samples = int(no_hit_value)
        evaluation_mode = "final_fallback"
        evaluation_step = int(final_step)
        evaluation_train_size = int(final_train_size)

    return {
        "reached_threshold": bool(reached_threshold),
        "first_hit_idx": None if first_hit_idx is None else int(first_hit_idx),
        "first_hit_step": None if first_hit_step is None else int(first_hit_step),
        "first_hit_samples": int(first_hit_samples),
        "evaluation_mode": evaluation_mode,
        "evaluation_step": int(evaluation_step),
        "evaluation_train_size": int(evaluation_train_size),
        "final_step": int(final_step),
        "final_train_size": int(final_train_size),
        "required_consecutive": int(REQUIRED_CONSECUTIVE),
        "evaluation_source": "recomputed_from_pf_model",
    }, None



def get_saved_posterior_metrics(output_data, step):
    pf_post_mean = get_saved_metric(output_data, "Pf_post_mean", step)
    pf_post_cov = get_saved_metric(output_data, "Pf_post_CoV", step)
    pf_post_ci95 = get_saved_metric(output_data, "Pf_post_CI95", step)

    if pf_post_mean is None or pf_post_cov is None or pf_post_ci95 is None:
        return None
    if not isinstance(pf_post_ci95, list) or len(pf_post_ci95) != 2:
        return None

    return {
        "Pf_post_mean": float(pf_post_mean),
        "Pf_post_CoV": float(pf_post_cov),
        "Pf_post_CI95_low": float(pf_post_ci95[0]),
        "Pf_post_CI95_high": float(pf_post_ci95[1]),
        "posterior_source": "saved_output",
        "model_source": "saved_output",
        "checkpoint_path": None,
    }



def estimate_posterior_from_model(
    model_gp,
    seed_exp,
    step,
    input_dim,
    n_g_pf,
    n_pf_post_pool,
    pf_post_batch_size,
    posterior_estimator,
    posterior_workers,
):
    pf_post_seed = derive_stream_seed(seed_exp, step, stream_id=1)
    pf_post_rng = np.random.RandomState(pf_post_seed)
    gp_cache = build_gp_cache_from_gpr(model_gp)
    pf_samples, pf_post_mean, pf_post_cov, pf_post_ci95 = estimate_pf_posterior_samples(
        cache=gp_cache,
        N_g=n_g_pf,
        batch_size_acq=pf_post_batch_size,
        rng=pf_post_rng,
        n_pool_pf=n_pf_post_pool,
        input_dim=input_dim,
        method=posterior_estimator,
        posterior_workers=posterior_workers,
        verbose=False,
    )
    return {
        "Pf_post_mean": float(pf_post_mean),
        "Pf_post_CoV": float(pf_post_cov),
        "Pf_post_CI95_low": float(pf_post_ci95[0]),
        "Pf_post_CI95_high": float(pf_post_ci95[1]),
        "pf_post_seed": int(pf_post_seed),
        "posterior_estimator": posterior_estimator,
        "pf_samples": np.asarray(pf_samples, dtype=np.float64),
    }



def resolve_predict_settings(config, args):
    if args.predict_n_jobs is not None:
        predict_n_jobs = resolve_cpu_workers(args.predict_n_jobs)
    elif "cpu_workers" in config:
        predict_n_jobs = resolve_cpu_workers(config.get("cpu_workers", -1))
    else:
        predict_n_jobs = resolve_cpu_workers(config.get("predict_n_jobs", DEFAULT_POST_PREDICT_N_JOBS))

    predict_batch_size = int(
        args.predict_batch_size
        if args.predict_batch_size is not None
        else config.get("predict_batch_size", DEFAULT_POST_PREDICT_BATCH_SIZE)
    )
    return int(predict_batch_size), int(predict_n_jobs)



def verify_pf_model_from_model(model_gp, output, config, lstate, evaluation_step, args):
    seed_exp = int(config.get("seed", 0))
    pf_model_seed = derive_stream_seed(seed_exp, evaluation_step, stream_id=2)
    n_mcs_pf = int(config["n_mcs_pf"])
    predict_batch_size, predict_n_jobs = resolve_predict_settings(config, args)

    np.random.seed(pf_model_seed)
    pf_model_replayed = estimate_pf_model_stream(
        model_gp,
        n_rows=n_mcs_pf,
        input_dim=int(lstate.input_dim),
        predict_n_jobs=predict_n_jobs,
        predict_batch_size=predict_batch_size,
        verbose=args.verbose,
    )
    pf_model_report = build_pf_model_report(
        get_saved_metric(output, "Pf_model", evaluation_step),
        pf_model_replayed,
        n_mcs_pf,
    )
    pf_model_report.update(
        {
            "pf_model_seed": int(pf_model_seed),
            "n_mcs_pf": int(n_mcs_pf),
            "predict_batch_size": int(predict_batch_size),
            "predict_n_jobs": int(predict_n_jobs),
            "verified": True,
        }
    )
    return pf_model_report



def empty_pf_model_report():
    return {
        "saved": None,
        "replayed": None,
        "abs_diff": None,
        "mc_se_saved": None,
        "mc_se_replayed": None,
        "mc_se_combined": None,
        "diff_over_combined_se": None,
        "pf_model_seed": None,
        "n_mcs_pf": None,
        "predict_batch_size": None,
        "predict_n_jobs": None,
        "verified": False,
    }



def evaluate_run(case, strategy, run_name, record, threshold_info, evaluation_info, args, lstate_cache, pf_post_samples_root):
    config = record["config"]
    output = record["output"]
    run_dir = record["run_dir"]

    evaluation_step = evaluation_info.get("evaluation_step")
    evaluation_train_size = evaluation_info.get("evaluation_train_size")
    if evaluation_step is None or evaluation_train_size is None:
        return None, "missing_evaluation_step"

    passive_samples = int(config.get("passive_samples", DOE_SAMPLES))
    al_batch = int(config.get("al_batch", 1))
    seed_exp = int(config.get("seed", 0))
    final_step = evaluation_info.get("final_step")
    final_train_size = evaluation_info.get("final_train_size")

    posterior_metrics = None
    if not args.force_recompute_posterior:
        posterior_metrics = get_saved_posterior_metrics(output, int(evaluation_step))
    posterior_source = None
    model_source = None
    checkpoint_path = None
    pf_post_samples_path = None
    pf_post_samples_count = None
    pf_model_report = empty_pf_model_report()

    if posterior_metrics is not None:
        posterior_source = posterior_metrics["posterior_source"]
        model_source = posterior_metrics["model_source"]
    else:
        x_all, y_all = load_training_samples(output)
        actual_final_train_size = int(x_all.shape[0])
        actual_final_step = int((actual_final_train_size - passive_samples) // al_batch)
        if evaluation_train_size > actual_final_train_size:
            return None, "evaluation_train_size_exceeds_history"

        if case not in lstate_cache:
            lstate_cache[case] = ls_REGISTRY[case]()
        lstate = lstate_cache[case]

        target_model, checkpoint_name, checkpoint_path_obj = try_load_exact_checkpoint(
            run_dir=run_dir,
            step=int(evaluation_step),
            target_train_size=int(evaluation_train_size),
            final_train_size=int(actual_final_train_size),
            verbose=args.verbose,
            allow_last_checkpoint_mismatch=(
                evaluation_info.get("evaluation_mode") == "final_fallback"
                and int(evaluation_step) == int(actual_final_step)
            ),
        )
        if target_model is not None:
            model_source = str(checkpoint_name)
            checkpoint_path = None if checkpoint_path_obj is None else str(checkpoint_path_obj)
        else:
            x_train = np.asarray(x_all[:evaluation_train_size], dtype=np.float64)
            y_train = np.asarray(y_all[:evaluation_train_size], dtype=np.float64)
            target_model, _ = fit_target_model(config, lstate, x_train, y_train)
            model_source = "direct_fit"

        if args.verify_pf_model:
            pf_model_report = verify_pf_model_from_model(
                model_gp=target_model,
                output=output,
                config=config,
                lstate=lstate,
                evaluation_step=int(evaluation_step),
                args=args,
            )

        n_g_pf = int(args.n_g_pf if args.n_g_pf is not None else config.get("n_g_pf", DEFAULT_POST_N_G_PF))
        n_pf_post_pool = int(
            args.n_pf_post_pool
            if args.n_pf_post_pool is not None
            else config.get("n_pf_post_pool", DEFAULT_POST_N_PF_POST_POOL)
        )
        pf_post_batch_size = int(
            args.pf_post_batch_size
            if args.pf_post_batch_size is not None
            else config.get("pf_post_batch_size", DEFAULT_POST_PF_POST_BATCH_SIZE)
        )
        posterior_metrics = estimate_posterior_from_model(
            model_gp=target_model,
            seed_exp=seed_exp,
            step=int(evaluation_step),
            input_dim=int(lstate.input_dim),
            n_g_pf=n_g_pf,
            n_pf_post_pool=n_pf_post_pool,
            pf_post_batch_size=pf_post_batch_size,
            posterior_estimator=args.posterior_estimator,
            posterior_workers=args.posterior_workers,
        )
        posterior_source = (
            "checkpoint_recomputed" if model_source != "direct_fit" else "direct_fit_recomputed"
        )
        final_train_size = int(actual_final_train_size)
        final_step = int(actual_final_step)
        if args.save_pf_post_samples:
            pf_samples = posterior_metrics.get("pf_samples")
            if pf_samples is not None:
                pf_post_samples_path_obj = pf_post_samples_path_for(
                    samples_root=pf_post_samples_root,
                    case=case,
                    strategy=strategy,
                    run_name=run_name,
                    evaluation_step=int(evaluation_step),
                    evaluation_train_size=int(evaluation_train_size),
                    posterior_estimator=str(posterior_metrics.get("posterior_estimator") or args.posterior_estimator),
                    n_g_pf=int(n_g_pf),
                    n_pf_post_pool=int(n_pf_post_pool),
                )
                save_pf_post_samples(
                    path=pf_post_samples_path_obj,
                    pf_samples=pf_samples,
                    metadata={
                        "case": np.asarray(case),
                        "strategy": np.asarray(strategy),
                        "run": np.asarray(run_name),
                        "evaluation_step": np.asarray(int(evaluation_step), dtype=np.int64),
                        "evaluation_train_size": np.asarray(int(evaluation_train_size), dtype=np.int64),
                        "pf_post_seed": np.asarray(int(posterior_metrics["pf_post_seed"]), dtype=np.int64),
                        "n_g_pf": np.asarray(int(n_g_pf), dtype=np.int64),
                        "n_pf_post_pool": np.asarray(int(n_pf_post_pool), dtype=np.int64),
                        "pf_post_batch_size": np.asarray(int(pf_post_batch_size), dtype=np.int64),
                        "posterior_workers": np.asarray(
                            -1 if args.posterior_workers is None else int(args.posterior_workers),
                            dtype=np.int64,
                        ),
                        "posterior_estimator": np.asarray(str(posterior_metrics.get("posterior_estimator") or args.posterior_estimator)),
                    },
                )
                pf_post_samples_path = str(pf_post_samples_path_obj)
                pf_post_samples_count = int(np.asarray(pf_samples).size)
        posterior_metrics.pop("pf_samples", None)

    n_g_pf_used = int(args.n_g_pf if args.n_g_pf is not None else config.get("n_g_pf", DEFAULT_POST_N_G_PF))
    n_pf_post_pool_used = int(
        args.n_pf_post_pool
        if args.n_pf_post_pool is not None
        else config.get("n_pf_post_pool", DEFAULT_POST_N_PF_POST_POOL)
    )
    pf_post_batch_size_used = int(
        args.pf_post_batch_size
        if args.pf_post_batch_size is not None
        else config.get("pf_post_batch_size", DEFAULT_POST_PF_POST_BATCH_SIZE)
    )

    threshold_defined_by = threshold_info.get("threshold_entry", {}).get("strategy")
    row = {
        "case": case,
        "case_title": CASE_TITLES.get(case, case),
        "strategy": strategy,
        "strategy_label": strategy_label(strategy),
        "run": run_name,
        "exp_num": parse_experiment_number(run_name),
        "seed": int(config.get("seed", 0)),
        "threshold_delta_pf": float(threshold_info["threshold_delta_pf"]),
        "threshold_defined_by": threshold_defined_by,
        "required_consecutive": int(threshold_info.get("required_consecutive", REQUIRED_CONSECUTIVE)),
        "threshold_source": threshold_info.get("threshold_source"),
        "evaluation_source": evaluation_info.get("evaluation_source"),
        "reached_threshold": bool(evaluation_info.get("reached_threshold")),
        "first_hit_idx": evaluation_info.get("first_hit_idx"),
        "first_hit_step": evaluation_info.get("first_hit_step"),
        "first_hit_samples": evaluation_info.get("first_hit_samples"),
        "evaluation_mode": evaluation_info.get("evaluation_mode"),
        "evaluation_step": int(evaluation_step),
        "evaluation_train_size": int(evaluation_train_size),
        "final_step": None if final_step is None else int(final_step),
        "final_train_size": None if final_train_size is None else int(final_train_size),
        "posterior_source": posterior_source,
        "model_source": model_source,
        "checkpoint_path": checkpoint_path,
        "posterior_estimator": posterior_metrics.get("posterior_estimator"),
        "force_recompute_posterior": bool(args.force_recompute_posterior),
        "n_g_pf": int(n_g_pf_used),
        "n_pf_post_pool": int(n_pf_post_pool_used),
        "pf_post_batch_size": int(pf_post_batch_size_used),
        "posterior_workers": None if args.posterior_workers is None else int(args.posterior_workers),
        "pf_post_seed": posterior_metrics.get("pf_post_seed"),
        "pf_post_samples_path": pf_post_samples_path,
        "pf_post_samples_count": pf_post_samples_count,
        "pf_post_mean": float(posterior_metrics["Pf_post_mean"]),
        "pf_post_cov": float(posterior_metrics["Pf_post_CoV"]),
        "pf_post_ci95_low": float(posterior_metrics["Pf_post_CI95_low"]),
        "pf_post_ci95_high": float(posterior_metrics["Pf_post_CI95_high"]),
        "pf_model_verified": bool(pf_model_report.get("verified", False)),
        "pf_model_saved": pf_model_report.get("saved"),
        "pf_model_replayed": pf_model_report.get("replayed"),
        "pf_model_abs_diff": pf_model_report.get("abs_diff"),
        "pf_model_mc_se_saved": pf_model_report.get("mc_se_saved"),
        "pf_model_mc_se_replayed": pf_model_report.get("mc_se_replayed"),
        "pf_model_mc_se_combined": pf_model_report.get("mc_se_combined"),
        "pf_model_diff_over_combined_se": pf_model_report.get("diff_over_combined_se"),
        "pf_model_seed": pf_model_report.get("pf_model_seed"),
        "n_mcs_pf": pf_model_report.get("n_mcs_pf"),
        "predict_batch_size": pf_model_report.get("predict_batch_size"),
        "predict_n_jobs": pf_model_report.get("predict_n_jobs"),
    }
    return row, None


def write_table(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "case_title",
        "strategy",
        "strategy_label",
        "run",
        "exp_num",
        "seed",
        "threshold_delta_pf",
        "threshold_defined_by",
        "required_consecutive",
        "threshold_source",
        "evaluation_source",
        "reached_threshold",
        "first_hit_idx",
        "first_hit_step",
        "first_hit_samples",
        "evaluation_mode",
        "evaluation_step",
        "evaluation_train_size",
        "final_step",
        "final_train_size",
        "posterior_source",
        "model_source",
        "checkpoint_path",
        "posterior_estimator",
        "force_recompute_posterior",
        "n_g_pf",
        "n_pf_post_pool",
        "pf_post_batch_size",
        "posterior_workers",
        "pf_post_seed",
        "pf_post_samples_path",
        "pf_post_samples_count",
        "pf_post_mean",
        "pf_post_cov",
        "pf_post_ci95_low",
        "pf_post_ci95_high",
        "pf_model_verified",
        "pf_model_saved",
        "pf_model_replayed",
        "pf_model_abs_diff",
        "pf_model_mc_se_saved",
        "pf_model_mc_se_replayed",
        "pf_model_mc_se_combined",
        "pf_model_diff_over_combined_se",
        "pf_model_seed",
        "n_mcs_pf",
        "predict_batch_size",
        "predict_n_jobs",
        "run_elapsed_sec",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f_id:
        writer = csv.DictWriter(f_id, fieldnames=fieldnames, delimiter="	")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_grouped_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["strategy"])
        grouped.setdefault(key, []).append(row)

    report("[grouped Pf_post_CoV summary]")
    for case in CASE_STUDIES:
        case_rows = {strategy: grouped[(case, strategy)] for strategy in DEFAULT_STRATEGIES if (case, strategy) in grouped}
        if not case_rows:
            continue

        report("=" * 90)
        report(f"CASE: {CASE_TITLES.get(case, case)}")
        report("=" * 90)
        report(
            f"{'Strategy':<16} | {'N':>3} | {'Hits':>4} | {'Fallback':>8} | "
            f"{'Saved':>5} | {'Chkpt':>5} | {'Direct':>6} | {'Mean':>10} | {'Median':>10} | {'P2.5':>10} | {'P97.5':>10}"
        )
        report("-" * 90)

        for strategy in DEFAULT_STRATEGIES:
            strategy_rows = case_rows.get(strategy)
            if not strategy_rows:
                continue

            cov_values = np.asarray([float(r["pf_post_cov"]) for r in strategy_rows], dtype=float)
            n_total = len(strategy_rows)
            n_hits = sum(1 for r in strategy_rows if r["evaluation_mode"] == "threshold_hit")
            n_fallback = sum(1 for r in strategy_rows if r["evaluation_mode"] == "final_fallback")
            n_saved = sum(1 for r in strategy_rows if r["posterior_source"] == "saved_output")
            n_chkpt = sum(1 for r in strategy_rows if r["posterior_source"] == "checkpoint_recomputed")
            n_direct = sum(1 for r in strategy_rows if r["posterior_source"] == "direct_fit_recomputed")
            p2_5, p97_5 = np.percentile(cov_values, [2.5, 97.5])

            report(
                f"{strategy_label(strategy):<16} | {n_total:>3d} | {n_hits:>4d} | {n_fallback:>8d} | "
                f"{n_saved:>5d} | {n_chkpt:>5d} | {n_direct:>6d} | "
                f"{np.mean(cov_values):>10.6e} | {np.median(cov_values):>10.6e} | "
                f"{p2_5:>10.6e} | {p97_5:>10.6e}"
            )
        report("")



def main():
    args = parse_args()
    if args.max_runs_per_strategy is not None and args.max_runs_per_strategy <= 0:
        raise ValueError("--max-runs-per-strategy must be positive when provided.")

    base_results_dir, aggregated_dir = resolve_results_dirs(args.results_folder)
    if not base_results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {base_results_dir}")

    threshold_json_path = aggregated_dir / THRESHOLD_DICT_NAME
    threshold_hits_path = aggregated_dir / THRESHOLD_HITS_TABLE_NAME
    strategy_rankings_path = aggregated_dir / STRATEGY_RANKINGS_TABLE_NAME
    table_path = aggregated_dir / PF_POST_COV_TABLE_NAME
    summary_path = aggregated_dir / PF_POST_COV_SUMMARY_NAME
    progress_path = progress_path_for(table_path)
    metadata_path = metadata_path_for(table_path)
    pf_post_samples_root = pf_post_samples_root_for(aggregated_dir)

    if args.overwrite:
        remove_if_exists(table_path)
        remove_if_exists(summary_path)

    campaign_settings = campaign_settings_dict(args, base_results_dir)
    rows, completed_keys, saved_metadata = resolve_resume_state(
        progress_path=progress_path,
        metadata_path=metadata_path,
        current_settings=campaign_settings,
        overwrite=bool(args.overwrite),
    )

    source_counts = {"saved_output": 0, "checkpoint_recomputed": 0, "direct_fit_recomputed": 0}
    eval_mode_counts = {"threshold_hit": 0, "final_fallback": 0}
    evaluation_source_counts = {"saved_threshold_hits": 0, "recomputed_from_pf_model": 0}
    pf_model_verified_count = 0
    for row in rows:
        update_counts_from_row(row, source_counts, eval_mode_counts, evaluation_source_counts)
        if row.get("pf_model_verified"):
            pf_model_verified_count += 1

    report("[start] Pf_post_CoV threshold postprocess")
    report(f"  BASE_RESULTS_DIR          : {base_results_dir}")
    report(f"  AGGREGATED_DIR            : {aggregated_dir}")
    report(f"  TABLE_PATH                : {table_path}")
    report(f"  PROGRESS_PATH             : {progress_path}")
    report(f"  METADATA_PATH             : {metadata_path}")
    report(f"  CASE_STUDIES              : {args.case_studies_filter or CASE_STUDIES}")
    report(f"  STRATEGIES                : {args.strategies_filter or DEFAULT_STRATEGIES}")
    report(f"  CAPTURED_LS               : {CAPTURED_LS}")
    report(f"  THRESHOLD_FACTOR          : {THRESHOLD_FACTOR}")
    report(f"  REQUIRED_CONSECUTIVE      : {REQUIRED_CONSECUTIVE}")
    report(f"  MAX_RUNS_PER_STRATEGY     : {args.max_runs_per_strategy}")
    report(f"  POSTERIOR_ESTIMATOR       : {args.posterior_estimator}")
    report(f"  POSTERIOR_WORKERS         : {args.posterior_workers}")
    report(f"  FORCE_RECOMPUTE_POSTERIOR : {args.force_recompute_posterior}")
    report(f"  SAVE_PF_POST_SAMPLES      : {args.save_pf_post_samples}")
    report(f"  PF_POST_SAMPLES_ROOT      : {pf_post_samples_root}")
    report(f"  DEFAULT_POST_N_G_PF       : {DEFAULT_POST_N_G_PF}")
    report(f"  DEFAULT_POST_N_PF_POOL    : {DEFAULT_POST_N_PF_POST_POOL}")
    report(f"  DEFAULT_POST_BATCH        : {DEFAULT_POST_PF_POST_BATCH_SIZE}")
    report(f"  VERIFY_PF_MODEL           : {args.verify_pf_model}")
    report(f"  RESUMED_ROWS              : {len(rows)}")
    report("")

    run_records, missing_files, runs_seen, runs_loaded = load_run_records(
        base_results_dir,
        max_runs_per_strategy=args.max_runs_per_strategy,
        case_studies_filter=args.case_studies_filter,
        strategies_filter=args.strategies_filter,
    )

    threshold_dict = load_threshold_json(threshold_json_path)
    threshold_source = str(threshold_json_path) if threshold_dict is not None else None
    threshold_fallback = False
    if threshold_dict is None:
        threshold_relative_error_dict, threshold_source, threshold_fallback = load_threshold_relative_error_dict(
            base_results_dir=base_results_dir,
            aggregated_dir=aggregated_dir,
            eval_run_records=run_records,
            max_runs_per_strategy=args.max_runs_per_strategy,
        )
        threshold_dict = build_threshold_dict(threshold_relative_error_dict)
        for case, data in threshold_dict.items():
            data["threshold_source"] = threshold_source
            data["required_consecutive"] = REQUIRED_CONSECUTIVE
    else:
        for case, data in threshold_dict.items():
            data["threshold_source"] = str(threshold_json_path)

    threshold_hit_rows = load_threshold_hit_rows(threshold_hits_path)
    threshold_hit_lookup = build_threshold_hit_lookup(threshold_hit_rows)

    report("[threshold artifacts]")
    report(f"  thresholds_by_case         : {threshold_source}")
    report(f"  threshold_hits_per_seed    : {threshold_hits_path if threshold_hit_rows else 'not available'}")
    report(f"  strategy_rankings          : {strategy_rankings_path}")
    if threshold_fallback:
        report("  note                       : threshold JSON missing, thresholds were rebuilt")
    if not threshold_hit_rows:
        report("  note                       : threshold-hit table missing, evaluation steps will be derived on the fly")
    report("")

    report("[thresholds]")
    for case, data in threshold_dict.items():
        report(
            f"  {case:<22} : {float(data['threshold_delta_pf']):.12e} "
            f"(defined by {strategy_label(data['threshold_entry']['strategy'])})"
        )
    report("")

    planned_jobs, skipped_reason_counts = plan_evaluations(
        run_records=run_records,
        threshold_dict=threshold_dict,
        threshold_hit_lookup=threshold_hit_lookup,
    )
    total_planned = len(planned_jobs)
    remaining_jobs = []
    for job in planned_jobs:
        evaluation_train_size = int(job["evaluation_info"]["evaluation_train_size"])
        row_key = build_row_key(job["case"], job["strategy"], job["run_name"], evaluation_train_size)
        job["row_key"] = row_key
        if row_key not in completed_keys:
            remaining_jobs.append(job)

    report("[campaign]")
    report(f"  planned_rows              : {total_planned}")
    report(f"  remaining_rows            : {len(remaining_jobs)}")
    report(f"  resumed_completed_rows    : {len(rows)}")
    report("")

    aggregated_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_table(sort_rows(rows), table_path)

    metadata = {
        "campaign_settings": campaign_settings,
        "table_path": str(table_path),
        "summary_path": str(summary_path),
        "progress_path": str(progress_path),
        "rows_planned": int(total_planned),
        "rows_completed": int(len(rows)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running" if remaining_jobs else "completed",
    }
    write_metadata(metadata_path, metadata)

    lstate_cache = {}
    loop_start = time.perf_counter()
    completed_this_session = 0

    for job_idx, job in enumerate(remaining_jobs, start=1):
        case = job["case"]
        strategy = job["strategy"]
        run_name = job["run_name"]
        evaluation_info = job["evaluation_info"]
        run_timer = time.perf_counter()

        row, skip_reason = evaluate_run(
            case=case,
            strategy=strategy,
            run_name=run_name,
            record=job["record"],
            threshold_info=job["threshold_info"],
            evaluation_info=evaluation_info,
            args=args,
            lstate_cache=lstate_cache,
            pf_post_samples_root=pf_post_samples_root,
        )
        run_elapsed = time.perf_counter() - run_timer

        if row is None:
            skipped_reason_counts[skip_reason] = skipped_reason_counts.get(skip_reason, 0) + 1
            progress(
                f"[skip][{len(rows)}/{total_planned}] {case}/{strategy}/{run_name} | "
                f"reason={skip_reason} | t={run_elapsed/60.0:.1f}m"
            )
            continue

        row["run_elapsed_sec"] = float(run_elapsed)
        rows.append(row)
        completed_keys.add(job["row_key"])
        completed_this_session += 1
        update_counts_from_row(row, source_counts, eval_mode_counts, evaluation_source_counts)
        if row["pf_model_verified"]:
            pf_model_verified_count += 1

        append_progress_row(progress_path, row)
        write_table(sort_rows(rows), table_path)

        total_elapsed = time.perf_counter() - loop_start
        avg_sec = total_elapsed / max(completed_this_session, 1)
        remaining_count = len(remaining_jobs) - job_idx
        eta_sec = avg_sec * remaining_count
        pf_model_msg = ""
        if row["pf_model_verified"] and row["pf_model_abs_diff"] is not None:
            pf_model_msg = f" | pf_diff={float(row['pf_model_abs_diff']):.3e}"
        samples_msg = ""
        if row.get("pf_post_samples_count") is not None:
            samples_msg = f" | pf_samples={int(row['pf_post_samples_count'])}"

        progress(
            f"[done][{len(rows)}/{total_planned}] {case}/{strategy}/{run_name} | "
            f"eval={row['evaluation_train_size']} | src={row['posterior_source']} | "
            f"est={row.get('posterior_estimator') or 'saved'} | mean={row['pf_post_mean']:.6e} | "
            f"cov={row['pf_post_cov']:.6e} | t={run_elapsed/60.0:.1f}m | eta={eta_sec/3600.0:.1f}h"
            f"{samples_msg}{pf_model_msg}"
        )

        if args.verbose:
            report(
                f"  [run] {case}/{strategy}/{run_name} | eval={row['evaluation_train_size']} | "
                f"mode={row['evaluation_mode']} | eval_src={row['evaluation_source']} | "
                f"post_src={row['posterior_source']} | est={row.get('posterior_estimator')} | "
                f"mean={row['pf_post_mean']:.6e} | cov={row['pf_post_cov']:.6e} | "
                f"elapsed={run_elapsed:.1f}s"
            )

        metadata.update(
            {
                "rows_completed": int(len(rows)),
                "last_completed_key": job["row_key"],
                "last_completed_case": case,
                "last_completed_strategy": strategy,
                "last_completed_run": run_name,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running" if remaining_count > 0 else "completed",
            }
        )
        write_metadata(metadata_path, metadata)

    rows_sorted = sort_rows(rows)
    write_table(rows_sorted, table_path)
    append_grouped_summary(rows_sorted)

    report("[summary]")
    report(f"  runs discovered           : {runs_seen}")
    report(f"  runs loaded               : {runs_loaded}")
    report(f"  planned rows              : {total_planned}")
    report(f"  resumed rows              : {len(completed_keys) - completed_this_session}")
    report(f"  rows written              : {len(rows_sorted)}")
    report(f"  threshold hits            : {eval_mode_counts.get('threshold_hit', 0)}")
    report(f"  final fallbacks           : {eval_mode_counts.get('final_fallback', 0)}")
    report(f"  evals from saved hits     : {evaluation_source_counts.get('saved_threshold_hits', 0)}")
    report(f"  evals from fallback       : {evaluation_source_counts.get('recomputed_from_pf_model', 0)}")
    report(f"  posterior from output     : {source_counts.get('saved_output', 0)}")
    report(f"  posterior from checkpoint : {source_counts.get('checkpoint_recomputed', 0)}")
    report(f"  posterior from direct fit : {source_counts.get('direct_fit_recomputed', 0)}")
    report(f"  Pf_model verified rows    : {pf_model_verified_count}")
    report(f"  missing file records      : {len(missing_files)}")
    if skipped_reason_counts:
        report("  skipped runs by reason    :")
        for reason, count in sorted(skipped_reason_counts.items()):
            report(f"    {reason}: {count}")
    report(f"[save][TABLE]    {table_path}")
    report(f"[save][PROGRESS] {progress_path}")
    report(f"[save][META]     {metadata_path}")

    metadata.update(
        {
            "rows_completed": int(len(rows_sorted)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
        }
    )
    write_metadata(metadata_path, metadata)
    flush_report_summary(summary_path)


if __name__ == "__main__":
    main()
