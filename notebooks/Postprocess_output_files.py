import json
import pickle
import numpy as np

from Postprocess_settings import (
    BASE_RESULTS_DIR,
    AGGREGATED_DIR,
    CASE_STUDIES,
    DEFAULT_STRATEGIES,
    GROUP_2D,
    GROUP_HD,
    CASE_TITLES,
    STRATEGY_DIR_MAP,
    STRATEGY_LABELS,
    STRATEGY_COLORS,
    DOE_SAMPLES,
    MAX_LEN_BY_CASE,
    REAL_PF_VALUES,
    strategy_to_dir,
)


# Keep this False by default because output dictionaries can be very large.
SAVE_OUTPUT_RESULTS_DICT = False

# If True, store Pf_model trajectory inside relative_error_dict entries.
INCLUDE_PF_MODEL_IN_RELATIVE = False


def build_relative_entry(output_data, pf_ref):
    pf_model_arr = np.asarray(output_data.get("Pf_model", []), dtype=float)
    if pf_model_arr.size == 0:
        return {
            "Pf_ref": pf_ref,
            "relative_error": [],
            "abs_relative_error": [],
            "final_relative_error": None,
            "final_abs_relative_error": None,
        }

    rel_error = (pf_model_arr - pf_ref) / pf_ref
    abs_rel_error = np.abs(rel_error)
    entry = {
        "Pf_ref": pf_ref,
        "relative_error": rel_error.tolist(),
        "abs_relative_error": abs_rel_error.tolist(),
        "final_relative_error": float(rel_error[-1]),
        "final_abs_relative_error": float(abs_rel_error[-1]),
    }
    if INCLUDE_PF_MODEL_IN_RELATIVE:
        entry["Pf_model"] = pf_model_arr.tolist()
    return entry


def build_metadata(pf_ref_by_case):
    return {
        "case_studies": CASE_STUDIES,
        "group_2d": GROUP_2D,
        "group_hd": GROUP_HD,
        "case_titles": CASE_TITLES,
        "default_strategies": DEFAULT_STRATEGIES,
        "strategy_dir_map": STRATEGY_DIR_MAP,
        "strategy_labels": STRATEGY_LABELS,
        "strategy_colors": STRATEGY_COLORS,
        "doe_samples": DOE_SAMPLES,
        "max_len_by_case": MAX_LEN_BY_CASE,
        "pf_reference_by_case": pf_ref_by_case,
        "save_output_results_dict": SAVE_OUTPUT_RESULTS_DICT,
        "include_pf_model_in_relative": INCLUDE_PF_MODEL_IN_RELATIVE,
    }


def main():
    if not BASE_RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Results directory not found: {BASE_RESULTS_DIR}")

    print(f"[start] BASE_RESULTS_DIR = {BASE_RESULTS_DIR}")
    print(f"[start] AGGREGATED_DIR   = {AGGREGATED_DIR}")
    print(f"[start] CASE_STUDIES     = {CASE_STUDIES}")
    print(f"[start] STRATEGIES       = {DEFAULT_STRATEGIES}")
    print(f"[start] SAVE_OUTPUT_RESULTS_DICT = {SAVE_OUTPUT_RESULTS_DICT}")
    print("")

    # Fixed reference probabilities used to replicate original figures.
    pf_ref_by_case = dict(REAL_PF_VALUES)

    output_results_dict = {}
    config_results_dict = {}
    relative_error_dict = {}
    missing_files = []

    runs_seen = 0
    runs_output_loaded = 0
    runs_config_loaded = 0
    runs_relative_loaded = 0

    for case in CASE_STUDIES:
        case_dir = BASE_RESULTS_DIR / case
        if not case_dir.is_dir():
            print(f"No directory found for case: {case}")
            continue

        print(f"[case] {case}")
        pf_ref = pf_ref_by_case.get(case, None)

        case_out = {}
        case_cfg = {}
        case_rel = {}

        for strategy in DEFAULT_STRATEGIES:
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
                print(f"  No directory found for {case}, {strategy}")
                continue

            run_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
            if len(run_dirs) == 0:
                print(f"  No run folders found for {case}, {strategy}")
                continue

            print(
                f"  [strategy] {strategy} -> {method_dir_name} | runs={len(run_dirs)}"
            )

            strategy_out = {}
            strategy_cfg = {}
            strategy_rel = {}

            for run_dir in run_dirs:
                runs_seen += 1
                run_name = run_dir.name
                output_path = run_dir / "output.json"
                config_path = run_dir / "config.json"

                if config_path.is_file():
                    with open(config_path, "r", encoding="utf-8") as f_id:
                        cfg_data = json.load(f_id)
                    strategy_cfg[run_name] = cfg_data
                    runs_config_loaded += 1
                else:
                    print(f"    No config.json found for {case}/{strategy}/{run_name}")
                    missing_files.append(
                        {
                            "case": case,
                            "strategy": strategy,
                            "run": run_name,
                            "missing": "config.json",
                        }
                    )

                if output_path.is_file():
                    with open(output_path, "r", encoding="utf-8") as f_id:
                        out_data = json.load(f_id)
                    strategy_out[run_name] = out_data
                    runs_output_loaded += 1

                    if pf_ref is not None and pf_ref > 0.0:
                        strategy_rel[run_name] = build_relative_entry(out_data, pf_ref)
                        runs_relative_loaded += 1
                else:
                    print(f"    No output.json found for {case}/{strategy}/{run_name}")
                    missing_files.append(
                        {
                            "case": case,
                            "strategy": strategy,
                            "run": run_name,
                            "missing": "output.json",
                        }
                    )

            if len(strategy_out) > 0:
                case_out[strategy] = strategy_out
            if len(strategy_cfg) > 0:
                case_cfg[strategy] = strategy_cfg
            if len(strategy_rel) > 0:
                case_rel[strategy] = strategy_rel

        if len(case_out) > 0:
            output_results_dict[case] = case_out
        if len(case_cfg) > 0:
            config_results_dict[case] = case_cfg
        if len(case_rel) > 0:
            relative_error_dict[case] = case_rel

        print("")

    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_OUTPUT_RESULTS_DICT:
        with open(AGGREGATED_DIR / "output_results_dict.pkl", "wb") as f_id:
            pickle.dump(output_results_dict, f_id)

    with open(AGGREGATED_DIR / "config_results_dict.pkl", "wb") as f_id:
        pickle.dump(config_results_dict, f_id)
    with open(AGGREGATED_DIR / "relative_error_dict.pkl", "wb") as f_id:
        pickle.dump(relative_error_dict, f_id)

    with open(AGGREGATED_DIR / "pf_reference_by_case.json", "w", encoding="utf-8") as f_id:
        json.dump(pf_ref_by_case, f_id, indent=2)
    with open(AGGREGATED_DIR / "missing_files.json", "w", encoding="utf-8") as f_id:
        json.dump(missing_files, f_id, indent=2)
    with open(AGGREGATED_DIR / "metadata.json", "w", encoding="utf-8") as f_id:
        json.dump(build_metadata(pf_ref_by_case), f_id, indent=2)

    n_cases = len(config_results_dict)
    n_method_groups = sum(len(v) for v in config_results_dict.values())
    n_runs_cfg = sum(
        len(runs) for methods in config_results_dict.values() for runs in methods.values()
    )
    n_runs_rel = sum(
        len(runs) for methods in relative_error_dict.values() for runs in methods.values()
    )

    print("[summary]")
    print(f"  cases with config               : {n_cases}")
    print(f"  method groups with config       : {n_method_groups}")
    print(f"  runs discovered                 : {runs_seen}")
    print(f"  runs with output.json           : {runs_output_loaded}")
    print(f"  runs with config.json           : {runs_config_loaded}")
    print(f"  runs with relative-error values : {runs_relative_loaded}")
    print(f"  runs counted in config dict     : {n_runs_cfg}")
    print(f"  runs counted in relative dict   : {n_runs_rel}")
    print(f"  missing file records            : {len(missing_files)}")
    print(f"[save] dictionaries written to: {AGGREGATED_DIR}")


if __name__ == "__main__":
    main()
