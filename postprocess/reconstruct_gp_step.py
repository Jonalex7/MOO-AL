import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from active_learning.eier import build_gp_cache_from_gpr, estimate_pf_posterior_samples
from limit_states import REGISTRY as ls_REGISTRY
from utils.data import parallel_predict, resolve_cpu_workers
from utils.gp_training import (
    fit_gp_with_optional_stabilization,
    make_base_kernel,
    resolve_gp_alpha,
)


DEFAULT_GP_ALPHA = 1e-8
DIRECT_FIT_RESTARTS = 9


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load an exact saved GP checkpoint when available, otherwise fit a GP once "
            "on the saved training samples at a chosen train size, then estimate Pf metrics "
            "and compare them to the values stored in output.json."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing config.json and output.json.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help=(
            "Total number of training samples to use, including passive_samples. "
            "Default: use the final saved train size from output.json."
        ),
    )
    parser.add_argument(
        "--n-mcs-pf",
        type=int,
        default=None,
        help="Override n_mcs_pf used for the Pf_model estimate.",
    )
    parser.add_argument(
        "--n-g-pf",
        type=int,
        default=None,
        help="Override n_g_pf used for Pf posterior sampling.",
    )
    parser.add_argument(
        "--n-pf-post-pool",
        type=int,
        default=None,
        help="Override n_pf_post_pool used for Pf posterior sampling.",
    )
    parser.add_argument(
        "--pf-post-batch-size",
        type=int,
        default=None,
        help="Override pf_post_batch_size used for Pf posterior sampling.",
    )
    parser.add_argument(
        "--posterior-estimator",
        choices=["dense", "streamed"],
        default="dense",
        help="Posterior Pf estimator to use for the replayed Pf_post_* metrics.",
    )
    parser.add_argument(
        "--posterior-workers",
        type=int,
        default=None,
        help="Override the number of worker threads used by the streamed posterior Pf estimator.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=None,
        help="Override predict_batch_size used for GP prediction batches.",
    )
    parser.add_argument(
        "--predict-n-jobs",
        type=int,
        default=None,
        help="Override predict_n_jobs used for GP predictions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model loading/fitting details and Pf prediction progress.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to save the report as JSON.",
    )
    return parser.parse_args()


def log(message: str, enabled: bool = True):
    if enabled:
        print(message)


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_run_data(run_dir: Path):
    config_path = run_dir / "config.json"
    output_path = run_dir / "output.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing output.json: {output_path}")

    with open(config_path, "r", encoding="utf-8") as f_id:
        config = json.load(f_id)
    with open(output_path, "r", encoding="utf-8") as f_id:
        output_data = json.load(f_id)
    return config_path, output_path, config, output_data


def load_training_samples(output_data):
    training_samples = output_data.get("training_samples")
    if not isinstance(training_samples, list) or len(training_samples) != 2:
        raise ValueError("output.json does not contain a valid training_samples field.")

    x_train = np.asarray(training_samples[0], dtype=np.float64)
    y_train = np.asarray(training_samples[1], dtype=np.float64).reshape(-1)

    if x_train.ndim != 2:
        raise ValueError("Saved x training samples must be a 2D array.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Saved x/y training samples have inconsistent lengths.")

    return x_train, y_train


def get_saved_metric(output_data, key, step):
    values = output_data.get(key)
    if not isinstance(values, list):
        return None
    if step < 0 or step >= len(values):
        return None
    return values[step]


def estimate_pf_model_stream(
    model_gp,
    n_rows,
    input_dim,
    predict_n_jobs,
    predict_batch_size,
    verbose=False,
):
    total_failures = 0
    total_rows = 0
    chunk_rows = max(int(predict_batch_size), int(predict_batch_size) * max(1, int(predict_n_jobs)))
    total_chunks = max(1, (int(n_rows) + chunk_rows - 1) // chunk_rows)
    progress_every = max(1, total_chunks // 10)

    log(
        (
            f"[predict] estimating Pf_model with n_mcs_pf={int(n_rows)} | "
            f"chunk_rows={chunk_rows} | total_chunks={total_chunks} | "
            f"predict_batch_size={int(predict_batch_size)} | predict_n_jobs={int(predict_n_jobs)}"
        ),
        enabled=True,
    )

    remaining = int(n_rows)
    chunk_idx = 0
    while remaining > 0:
        current = min(remaining, chunk_rows)
        x_batch = np.random.normal(0.0, 1.0, size=(current, input_dim))
        mean_pf, _ = parallel_predict(
            model_gp,
            x_batch,
            n_jobs=predict_n_jobs,
            batch_size=predict_batch_size,
        )
        total_failures += int(np.count_nonzero(mean_pf < 0.0))
        total_rows += int(current)
        remaining -= current
        chunk_idx += 1

        if verbose and (
            chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % progress_every == 0
        ):
            log(
                (
                    f"[predict] chunk {chunk_idx:>4d}/{total_chunks} | "
                    f"processed_rows={total_rows} | current_pf={total_failures / max(total_rows, 1):.6e}"
                ),
                enabled=True,
            )

    return float(total_failures / max(total_rows, 1))


def build_report_value(saved_value, replayed_value):
    report = {
        "saved": saved_value,
        "replayed": replayed_value,
    }
    if saved_value is None or replayed_value is None:
        report["abs_diff"] = None
        return report

    saved_arr = np.asarray(saved_value, dtype=np.float64)
    replayed_arr = np.asarray(replayed_value, dtype=np.float64)
    report["abs_diff"] = np.abs(replayed_arr - saved_arr).tolist()
    if saved_arr.ndim == 0:
        report["abs_diff"] = float(report["abs_diff"])
    return report


def build_pf_model_report(saved_value, replayed_value, n_mcs_pf):
    report = build_report_value(saved_value, replayed_value)

    if saved_value is not None:
        saved_float = float(saved_value)
        report["mc_se_saved"] = float(np.sqrt(saved_float * max(1.0 - saved_float, 0.0) / int(n_mcs_pf)))
    else:
        report["mc_se_saved"] = None

    if replayed_value is not None:
        replayed_float = float(replayed_value)
        report["mc_se_replayed"] = float(
            np.sqrt(replayed_float * max(1.0 - replayed_float, 0.0) / int(n_mcs_pf))
        )
    else:
        report["mc_se_replayed"] = None

    if report["mc_se_saved"] is not None and report["mc_se_replayed"] is not None and report["abs_diff"] is not None:
        combined_se = float(
            np.sqrt(report["mc_se_saved"] ** 2 + report["mc_se_replayed"] ** 2)
        )
        report["mc_se_combined"] = combined_se
        report["diff_over_combined_se"] = float(report["abs_diff"] / combined_se) if combined_se > 0 else None
    else:
        report["mc_se_combined"] = None
        report["diff_over_combined_se"] = None

    return report


def fmt_scalar(value):
    if value is None:
        return "None"
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    return f"{value:.6e}"


def normalize_gp_fit_info(gp_fit_info):
    if gp_fit_info is None:
        return {
            "stabilized": None,
            "alpha_used": None,
            "retry_count": None,
            "kernel_upper": None,
            "n_restarts_optimizer": None,
        }

    return {
        "stabilized": bool(gp_fit_info["stabilized"]),
        "alpha_used": float(gp_fit_info["alpha_used"]),
        "retry_count": int(gp_fit_info["retry_count"]),
        "kernel_upper": float(gp_fit_info["kernel_upper"]),
        "n_restarts_optimizer": int(gp_fit_info.get("n_restarts_optimizer", DIRECT_FIT_RESTARTS)),
    }


def resolve_target(config, x_all, args):
    passive_samples = int(config["passive_samples"])
    al_batch = int(config["al_batch"])
    available_train_size = int(len(x_all))

    delta_final = available_train_size - passive_samples
    if delta_final < 0:
        raise ValueError(
            f"Saved training history ({available_train_size}) is smaller than passive_samples={passive_samples}."
        )
    if delta_final % al_batch != 0:
        raise ValueError(
            f"Saved train size {available_train_size} is incompatible with passive_samples={passive_samples} "
            f"and al_batch={al_batch}."
        )

    final_step = delta_final // al_batch

    if args.train_size is None:
        train_size = available_train_size
        step = final_step
    else:
        train_size = int(args.train_size)
        if train_size < passive_samples:
            raise ValueError(
                f"Requested train_size {train_size} is smaller than passive_samples={passive_samples}."
            )
        delta = train_size - passive_samples
        if delta % al_batch != 0:
            raise ValueError(
                f"Requested train_size {train_size} is incompatible with passive_samples={passive_samples} "
                f"and al_batch={al_batch}. Expected passive_samples + k * al_batch."
            )
        step = delta // al_batch

    if train_size > available_train_size:
        raise ValueError(
            f"Requested train_size {train_size} exceeds saved training history of {available_train_size}."
        )
    if step > final_step:
        raise ValueError(
            f"Requested step {step} exceeds available training history. Maximum reachable step is {final_step}."
        )

    return int(step), int(train_size), int(final_step), int(available_train_size)


def build_doe_check(config, x_all, y_all, lstate):
    passive_samples = int(config["passive_samples"])
    if passive_samples == 0:
        return {
            "x_match": True,
            "y_match": True,
            "x_max_abs_diff": 0.0,
            "y_max_abs_diff": 0.0,
        }

    seed_exp = int(config["seed"])
    random_state = np.random.RandomState(seed_exp)
    x_doe_replayed, _, y_doe_replayed = lstate.get_doe(
        n_samples=passive_samples,
        method="lhs",
        random_state=random_state,
    )
    x_doe_saved = np.asarray(x_all[:passive_samples], dtype=np.float64)
    y_doe_saved = np.asarray(y_all[:passive_samples], dtype=np.float64)
    x_doe_replayed = np.asarray(x_doe_replayed, dtype=np.float64)
    y_doe_replayed = np.asarray(y_doe_replayed, dtype=np.float64)

    return {
        "x_match": bool(np.allclose(x_doe_saved, x_doe_replayed)),
        "y_match": bool(np.allclose(y_doe_saved, y_doe_replayed)),
        "x_max_abs_diff": float(np.max(np.abs(x_doe_saved - x_doe_replayed))),
        "y_max_abs_diff": float(np.max(np.abs(y_doe_saved - y_doe_replayed))),
    }


def derive_stream_seed(seed_exp, step, stream_id):
    seed_seq = np.random.SeedSequence([int(seed_exp), int(step), int(stream_id)])
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def resolve_checkpoint_path(run_dir, step, target_train_size, final_train_size):
    model_dir = run_dir / "model"
    if not model_dir.is_dir():
        return None, None

    candidates = []
    if target_train_size == final_train_size:
        candidates.append(("gp_last", model_dir / "gp_last.pkl"))
    candidates.append((f"gp_{step}", model_dir / f"gp_{step}.pkl"))

    seen_paths = set()
    for checkpoint_name, checkpoint_path in candidates:
        checkpoint_key = str(checkpoint_path)
        if checkpoint_key in seen_paths:
            continue
        seen_paths.add(checkpoint_key)
        if checkpoint_path.is_file():
            return checkpoint_name, checkpoint_path

    return None, None


def try_load_exact_checkpoint(run_dir, step, target_train_size, final_train_size, verbose, allow_last_checkpoint_mismatch=False):
    checkpoint_name, checkpoint_path = resolve_checkpoint_path(
        run_dir=run_dir,
        step=step,
        target_train_size=target_train_size,
        final_train_size=final_train_size,
    )
    if checkpoint_path is None:
        return None, None, None

    log(f"[model] loading exact checkpoint: {checkpoint_path}", enabled=True)
    try:
        with open(checkpoint_path, "rb") as f_id:
            model_gp = pickle.load(f_id)
    except Exception as exc:
        log(
            f"[model] failed to load checkpoint {checkpoint_path}: {exc}. Falling back to direct fit.",
            enabled=True,
        )
        return None, None, None

    model_train_inputs = getattr(model_gp, "X_train_", None)
    if model_train_inputs is None:
        log(
            f"[model] checkpoint {checkpoint_path} has no X_train_ attribute. Falling back to direct fit.",
            enabled=True,
        )
        return None, None, None

    checkpoint_train_size = int(np.asarray(model_train_inputs).shape[0])
    if checkpoint_train_size != int(target_train_size):
        if checkpoint_name == "gp_last" and allow_last_checkpoint_mismatch:
            log(
                (
                    f"[model] gp_last.pkl was trained on {checkpoint_train_size} samples while "
                    f"training_samples stores {target_train_size}. Using gp_last anyway because "
                    "the default target is the final saved model."
                ),
                enabled=True,
            )
        else:
            log(
                (
                    f"[model] checkpoint {checkpoint_path.name} was trained on {checkpoint_train_size} samples, "
                    f"not the requested {target_train_size}. Falling back to direct fit."
                ),
                enabled=True,
            )
            return None, None, None

    if verbose:
        log(
            f"[model] exact checkpoint accepted with train_size={checkpoint_train_size}.",
            enabled=True,
        )
    return model_gp, checkpoint_name, checkpoint_path


def fit_target_model(config, lstate, x_train, y_train):
    al_strategy = config["al_strategy"]
    gp_alpha = DEFAULT_GP_ALPHA
    if al_strategy == "eier" and "obs_stddev" in config:
        gp_alpha = resolve_gp_alpha(config, y_train)

    init_kernel = make_base_kernel(lstate.input_dim)
    model_gp, gp_fit_info = fit_gp_with_optional_stabilization(
        x_train_norm=x_train,
        y_train=y_train,
        init_kernel=init_kernel,
        input_dim=lstate.input_dim,
        gp_alpha=gp_alpha,
        n_restarts_optimizer=DIRECT_FIT_RESTARTS,
        enable_stabilization=(al_strategy == "eier"),
    )
    gp_fit_info = dict(gp_fit_info)
    gp_fit_info["n_restarts_optimizer"] = DIRECT_FIT_RESTARTS
    return model_gp, gp_fit_info


def evaluate_target(run_dir, config, output_data, args):
    x_all, y_all = load_training_samples(output_data)

    passive_samples = int(config["passive_samples"])
    seed_exp = int(config["seed"])
    case_study = config["case_study"]

    step, target_train_size, final_step, final_train_size = resolve_target(config, x_all, args)

    lstate = ls_REGISTRY[case_study]()
    if x_all.shape[1] != int(lstate.input_dim):
        raise ValueError(
            f"Saved training input dimension {x_all.shape[1]} does not match limit state input_dim {lstate.input_dim}."
        )

    n_mcs_pf = int(args.n_mcs_pf if args.n_mcs_pf is not None else config["n_mcs_pf"])
    n_g_pf = int(args.n_g_pf if args.n_g_pf is not None else config.get("n_g_pf", 1000))
    n_pf_post_pool = int(
        args.n_pf_post_pool if args.n_pf_post_pool is not None else config.get("n_pf_post_pool", 10000)
    )
    pf_post_batch_size = int(
        args.pf_post_batch_size if args.pf_post_batch_size is not None else config.get("pf_post_batch_size", 500)
    )
    predict_batch_size = int(
        args.predict_batch_size if args.predict_batch_size is not None else config.get("predict_batch_size", 10000)
    )

    if args.predict_n_jobs is not None:
        predict_n_jobs = resolve_cpu_workers(args.predict_n_jobs)
    else:
        raw_cpu_workers = config.get("cpu_workers", None)
        if raw_cpu_workers is not None:
            predict_n_jobs = resolve_cpu_workers(raw_cpu_workers)
        else:
            predict_n_jobs = resolve_cpu_workers(config.get("predict_n_jobs", -1))

    log(f"[setup] Using saved seed={seed_exp} from run config.", enabled=True)
    log(
        (
            f"[target] passive_samples={passive_samples}, requested_train_size={target_train_size}, "
            f"step={step}, final_saved_train_size={final_train_size}, final_step={final_step}"
        ),
        enabled=True,
    )
    log(
        (
            f"[target] estimate settings: n_mcs_pf={n_mcs_pf}, n_g_pf={n_g_pf}, "
            f"n_pf_post_pool={n_pf_post_pool}, pf_post_batch_size={pf_post_batch_size}, "
            f"predict_batch_size={predict_batch_size}, predict_n_jobs={predict_n_jobs}, "
            f"posterior_estimator={args.posterior_estimator}"
        ),
        enabled=True,
    )

    doe_check = build_doe_check(config, x_all, y_all, lstate)

    x_train = np.asarray(x_all[:target_train_size], dtype=np.float64)
    y_train = np.asarray(y_all[:target_train_size], dtype=np.float64)

    target_model, checkpoint_name, checkpoint_path = try_load_exact_checkpoint(
        run_dir=run_dir,
        step=step,
        target_train_size=target_train_size,
        final_train_size=final_train_size,
        verbose=args.verbose,
        allow_last_checkpoint_mismatch=(args.train_size is None),
    )

    model_source = None
    gp_fit_info = None
    if target_model is not None:
        model_source = checkpoint_name
        log(f"[model] using saved model source={model_source}", enabled=True)
    else:
        log(
            (
                f"[model] no exact checkpoint available for train_size={target_train_size}. "
                "Fitting GP directly on the saved training subset."
            ),
            enabled=True,
        )
        target_model, gp_fit_info = fit_target_model(config, lstate, x_train, y_train)
        model_source = "direct_fit"

    target_lml_raw = getattr(target_model, "log_marginal_likelihood_value_", None)
    target_lml = None if target_lml_raw is None else float(target_lml_raw)
    model_kernel = str(getattr(target_model, "kernel_", None))

    pf_post_seed = derive_stream_seed(seed_exp, step, stream_id=1)
    pf_model_seed = derive_stream_seed(seed_exp, step, stream_id=2)
    log(
        (
            f"[seed] derived prediction seeds from run seed: "
            f"pf_post_seed={pf_post_seed}, pf_model_seed={pf_model_seed}"
        ),
        enabled=True,
    )

    log(
        (
            f"[posterior] step={step} | estimating posterior Pf metrics with seed={pf_post_seed}, "
            f"n_g_pf={n_g_pf}, n_pf_post_pool={n_pf_post_pool}, batch_size={pf_post_batch_size}, "
            f"estimator={args.posterior_estimator}, posterior_workers={args.posterior_workers}"
        ),
        enabled=True,
    )
    pf_post_rng = np.random.RandomState(pf_post_seed)
    gp_cache = build_gp_cache_from_gpr(target_model)
    _, pf_post_mean, pf_post_cov, pf_post_ci95 = estimate_pf_posterior_samples(
        cache=gp_cache,
        N_g=n_g_pf,
        batch_size_acq=pf_post_batch_size,
        rng=pf_post_rng,
        n_pool_pf=n_pf_post_pool,
        input_dim=lstate.input_dim,
        method=args.posterior_estimator,
        posterior_workers=args.posterior_workers,
        verbose=args.verbose,
    )

    np.random.seed(pf_model_seed)
    pf_model = estimate_pf_model_stream(
        model_gp=target_model,
        n_rows=n_mcs_pf,
        input_dim=lstate.input_dim,
        predict_n_jobs=predict_n_jobs,
        predict_batch_size=predict_batch_size,
        verbose=args.verbose,
    )

    report = {
        "case_study": case_study,
        "step": int(step),
        "train_size": int(target_train_size),
        "final_train_size": int(final_train_size),
        "settings": {
            "seed": int(seed_exp),
            "pf_post_seed": int(pf_post_seed),
            "pf_model_seed": int(pf_model_seed),
            "n_mcs_pf": int(n_mcs_pf),
            "n_g_pf": int(n_g_pf),
            "n_pf_post_pool": int(n_pf_post_pool),
            "pf_post_batch_size": int(pf_post_batch_size),
            "predict_batch_size": int(predict_batch_size),
            "predict_n_jobs": int(predict_n_jobs),
            "posterior_estimator": args.posterior_estimator,
            "posterior_workers": args.posterior_workers,
        },
        "doe_replay_check": doe_check,
        "model": {
            "source": model_source,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "kernel": model_kernel,
            "model_train_size": int(np.asarray(target_model.X_train_).shape[0]),
        },
        "gp_fit_info": normalize_gp_fit_info(gp_fit_info),
        "metrics": {
            "Pf_model": build_pf_model_report(
                get_saved_metric(output_data, "Pf_model", step),
                pf_model,
                n_mcs_pf,
            ),
            "lml": build_report_value(get_saved_metric(output_data, "lml", step), target_lml),
            "Pf_post_mean": build_report_value(get_saved_metric(output_data, "Pf_post_mean", step), float(pf_post_mean)),
            "Pf_post_CoV": build_report_value(get_saved_metric(output_data, "Pf_post_CoV", step), float(pf_post_cov)),
            "Pf_post_CI95": build_report_value(
                get_saved_metric(output_data, "Pf_post_CI95", step),
                [float(pf_post_ci95[0]), float(pf_post_ci95[1])],
            ),
        },
    }
    return report


def print_report(run_dir: Path, report):
    print(f"Run directory           : {run_dir}")
    print(f"Case study              : {report['case_study']}")
    print(f"Step                    : {report['step']}")
    print(f"Train size              : {report['train_size']}")
    print(f"Final saved train size  : {report['final_train_size']}")
    print(f"Model source            : {report['model']['source']}")
    if report["model"]["checkpoint_path"] is not None:
        print(f"Checkpoint path         : {report['model']['checkpoint_path']}")
    print(
        "Estimate settings      : "
        f"seed={report['settings']['seed']}, "
        f"pf_post_seed={report['settings']['pf_post_seed']}, "
        f"pf_model_seed={report['settings']['pf_model_seed']}, "
        f"n_mcs_pf={report['settings']['n_mcs_pf']}, "
        f"n_g_pf={report['settings']['n_g_pf']}, "
        f"n_pf_post_pool={report['settings']['n_pf_post_pool']}, "
        f"posterior_estimator={report['settings']['posterior_estimator']}, "
        f"posterior_workers={report['settings']['posterior_workers']}"
    )
    print(
        "DoE replay check       : "
        f"x_match={report['doe_replay_check']['x_match']} "
        f"(max_abs={fmt_scalar(report['doe_replay_check']['x_max_abs_diff'])}), "
        f"y_match={report['doe_replay_check']['y_match']} "
        f"(max_abs={fmt_scalar(report['doe_replay_check']['y_max_abs_diff'])})"
    )
    if report["gp_fit_info"]["stabilized"] is None:
        print("GP fit info            : not available (loaded saved model)")
    else:
        print(
            "GP fit info            : "
            f"stabilized={report['gp_fit_info']['stabilized']}, "
            f"alpha_used={fmt_scalar(report['gp_fit_info']['alpha_used'])}, "
            f"retry_count={report['gp_fit_info']['retry_count']}, "
            f"kernel_upper={fmt_scalar(report['gp_fit_info']['kernel_upper'])}, "
            f"restarts={report['gp_fit_info']['n_restarts_optimizer']}"
        )
    print(f"Kernel                 : {report['model']['kernel']}")
    print(f"Model train size       : {report['model']['model_train_size']}")
    print("Metrics:")
    for key, metric_report in report["metrics"].items():
        print(f"  {key}:")
        print(f"    saved     = {metric_report['saved']}")
        print(f"    replayed  = {metric_report['replayed']}")
        print(f"    abs_diff  = {metric_report['abs_diff']}")
        if key == "Pf_model":
            print(f"    mc_se_saved         = {metric_report['mc_se_saved']}")
            print(f"    mc_se_replayed      = {metric_report['mc_se_replayed']}")
            print(f"    mc_se_combined      = {metric_report['mc_se_combined']}")
            print(f"    diff/combined_se    = {metric_report['diff_over_combined_se']}")


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    config_path, output_path, config, output_data = load_run_data(run_dir)
    log(f"[load] config.json: {config_path}", enabled=True)
    log(f"[load] output.json: {output_path}", enabled=True)
    report = evaluate_target(run_dir, config, output_data, args)
    report["run_dir"] = str(run_dir)
    report["config_path"] = str(config_path)
    report["output_path"] = str(output_path)
    print_report(run_dir, report)

    if args.report_json is not None:
        report_path = resolve_path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f_id:
            json.dump(report, f_id, indent=2)
        print(f"Report written to       : {report_path}")


if __name__ == "__main__":
    main()
