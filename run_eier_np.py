import argparse
import json
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from limit_states import REGISTRY as ls_REGISTRY
from utils.data import custom_optimizer, isoprobabilistic_transform

# ---------------------------------------------------------------------------
# Precision constants
# ---------------------------------------------------------------------------
TRAIN_DTYPE = np.float64
PRED_DTYPE = np.float64

# ---------------------------------------------------------------------------
# GP training (scikit-learn)
# ---------------------------------------------------------------------------
def train_gp_numpy(
    X_train: np.ndarray,
    y_train_norm: np.ndarray,
    dim: int,
    obs_stddev_norm: float,
    n_restarts: int = 9,
) -> GaussianProcessRegressor:
    """Fit an ARD Matern-5/2 GP on externally normalized outputs.

    We keep normalization explicit (instead of normalize_y=True) to control
    jitter scaling consistently in normalized space and keep the look-ahead
    algebra (mu/std in physical units) fully transparent.
    """

    kernel = ConstantKernel(1.0, (1e-6, 1e6)) * Matern(
        length_scale=np.ones(dim, dtype=TRAIN_DTYPE),
        length_scale_bounds=(1e-6, 1e6),
        nu=2.5,
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(obs_stddev_norm ** 2),
        optimizer=custom_optimizer,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,  # y is already normalized externally
        copy_X_train=False,
    )
    gpr.fit(X_train, y_train_norm.ravel())
    return gpr


# ---------------------------------------------------------------------------
# Matern-5/2 kernel (pure NumPy)
# ---------------------------------------------------------------------------
def _matern52_cross(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: np.ndarray,
    variance: float,
) -> np.ndarray:
    """K(X1, X2) for ARD Matern-5/2."""
    X1s = X1 / lengthscale
    X2s = X2 / lengthscale
    sq = (
        np.sum(X1s ** 2, axis=1, keepdims=True)
        + np.sum(X2s ** 2, axis=1)
        - 2.0 * X1s.dot(X2s.T)
    )
    r2 = np.maximum(sq, 0.0)
    r = np.sqrt(r2 + 1e-20)
    sqrt5_r = np.sqrt(5.0) * r
    return variance * (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * np.exp(-sqrt5_r)


# ---------------------------------------------------------------------------
# GP cache (fixed-shape like JAX version)
# ---------------------------------------------------------------------------
class GPCache(NamedTuple):
    X_train: np.ndarray     # (budget, dim)
    alpha: np.ndarray       # (budget, 1), normalized-output alpha
    Kinv: np.ndarray        # (budget, budget), normalized-output K^{-1}
    lengthscale: np.ndarray # (dim,)
    kern_var: float         # scalar (normalized-output kernel variance)
    y_mean: float           # output mean in physical units
    y_std: float            # output std in physical units
    n_train: int            # scalar


def _compute_y_scaler(y_train: np.ndarray):
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_std = np.maximum(y_std, np.asarray(1e-12, dtype=y_train.dtype))
    return y_mean, y_std


def build_gp_cache_numpy(
    gpr: GaussianProcessRegressor,
    Xbuf: np.ndarray,
    Ybuf: np.ndarray,
    n_train: int,
    budget: int,
    obs_stddev_norm: float,
    y_mean: float,
    y_std: float,
) -> GPCache:
    """Extract hyperparameters and build alpha/Kinv in normalized output space."""

    kernel_ = gpr.kernel_
    if hasattr(kernel_, "k1") and hasattr(kernel_, "k2"):
        kern_var = float(kernel_.k1.constant_value)
        lengthscale = np.asarray(kernel_.k2.length_scale, dtype=PRED_DTYPE)
    else:
        raise RuntimeError(f"Unexpected fitted kernel format: {kernel_!r}")

    X_valid = Xbuf[:n_train]
    y_valid_norm = ((Ybuf[:n_train] - y_mean) / y_std).astype(TRAIN_DTYPE)

    K_nn = _matern52_cross(X_valid, X_valid, lengthscale, kern_var)
    K_nn = K_nn + (obs_stddev_norm ** 2) * np.eye(n_train, dtype=TRAIN_DTYPE)

    alpha = np.linalg.solve(K_nn, y_valid_norm)  # (n,1)
    Kinv = np.linalg.solve(K_nn, np.eye(n_train, dtype=TRAIN_DTYPE))

    alpha_pad = np.zeros((budget, 1), dtype=TRAIN_DTYPE)
    Kinv_pad = np.zeros((budget, budget), dtype=TRAIN_DTYPE)
    alpha_pad[:n_train, :] = alpha
    Kinv_pad[:n_train, :n_train] = Kinv

    return GPCache(
        X_train=Xbuf.astype(PRED_DTYPE),
        alpha=alpha_pad.astype(PRED_DTYPE),
        Kinv=Kinv_pad.astype(PRED_DTYPE),
        lengthscale=lengthscale.astype(PRED_DTYPE),
        kern_var=np.float64(kern_var),
        y_mean=np.float64(y_mean),
        y_std=np.float64(y_std),
        n_train=int(n_train),
    )


# ---------------------------------------------------------------------------
# Prediction and uncertainty helpers
# ---------------------------------------------------------------------------
def _predict_mu_std(cache: GPCache, X_test: np.ndarray):
    K_star = _matern52_cross(X_test, cache.X_train, cache.lengthscale, cache.kern_var)
    mu_norm = K_star.dot(cache.alpha)
    v = K_star.dot(cache.Kinv)
    var_norm = cache.kern_var - np.sum(v * K_star, axis=1, keepdims=True)
    std_norm = np.sqrt(np.maximum(var_norm, np.asarray(1e-12, dtype=PRED_DTYPE)))

    mu = cache.y_mean + cache.y_std * mu_norm
    std = cache.y_std * std_norm
    return mu, std


def misclass_prob(mu: np.ndarray, std: np.ndarray):
    """tau_n(x) = Phi(-|mu|/std)."""
    z = -np.abs(mu) / np.maximum(std, 1e-12)
    return norm.cdf(z)


def _posterior_cov_cross(cache: GPCache, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    """Posterior covariance cross-covariance Cov(g(XA), g(XB) | D)."""
    K_ab = _matern52_cross(XA, XB, cache.lengthscale, cache.kern_var)
    K_ax = _matern52_cross(XA, cache.X_train, cache.lengthscale, cache.kern_var)
    K_bx = _matern52_cross(XB, cache.X_train, cache.lengthscale, cache.kern_var)
    return K_ab - (K_ax.dot(cache.Kinv)).dot(K_bx.T)


def estimate_pf_posterior_samples(
    cache: GPCache,
    X_pool_fixed: np.ndarray,
    N_g: int,
    batch_size_acq: int,
    rng: np.random.RandomState,
):
    """Posterior pf samples using a low-rank GP trajectory approximation on S.

    Returns
    -------
    pf_samples : np.ndarray, shape (N_g,)
    pf_mean : float
    pf_cov : float
    ci95 : tuple[float, float]
    """
    N = int(X_pool_fixed.shape[0])
    N_g = int(N_g)
    bs = int(batch_size_acq)
    if N <= 0 or N_g <= 0:
        raise ValueError("X_pool_fixed and N_g must be positive.")

    # Inducing subset size for low-rank posterior trajectory sampling.
    M = min(bs, N)
    idx_ind = rng.choice(N, size=M, replace=False)
    X_ind = X_pool_fixed[idx_ind]

    # Posterior mean on S, evaluated in batches.
    mu_pool = np.zeros((N,), dtype=PRED_DTYPE)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        mu_b, _ = _predict_mu_std(cache, X_pool_fixed[s:e])
        mu_pool[s:e] = mu_b.reshape(-1)

    # Build Cov(S, I) in batches and Cov(I, I) exactly.
    C_SI = np.zeros((N, M), dtype=PRED_DTYPE)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        C_SI[s:e] = _posterior_cov_cross(cache, X_pool_fixed[s:e], X_ind)

    C_II = _posterior_cov_cross(cache, X_ind, X_ind).astype(PRED_DTYPE)
    C_II = 0.5 * (C_II + C_II.T)

    # Stable factorization.
    jitter = 1e-10
    eye_M = np.eye(M, dtype=PRED_DTYPE)
    L_II = None
    for _ in range(6):
        try:
            L_II = np.linalg.cholesky(C_II + jitter * eye_M)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if L_II is None:
        raise np.linalg.LinAlgError("Failed Cholesky on inducing posterior covariance.")

    # Cov(S,S) ≈ B B^T with B = Cov(S,I) * Cov(I,I)^(-1/2) = Cov(S,I) * L_II^{-T}
    B = np.linalg.solve(L_II, C_SI.T).T  # (N, M)

    pf_samples = np.zeros((N_g,), dtype=PRED_DTYPE)
    g_chunk = max(1, min(N_g, bs))
    for g0 in range(0, N_g, g_chunk):
        g1 = min(N_g, g0 + g_chunk)
        eps = rng.normal(size=(M, g1 - g0)).astype(PRED_DTYPE)
        g_draws = mu_pool[:, None] + B.dot(eps)  # (N, g_chunk)
        pf_samples[g0:g1] = np.mean(g_draws < 0.0, axis=0)

    pf_mean = float(np.mean(pf_samples))
    pf_std = float(np.std(pf_samples, ddof=1)) if N_g > 1 else 0.0
    pf_cov = float(pf_std / max(pf_mean, 1e-16))
    ci95 = (float(np.quantile(pf_samples, 0.025)), float(np.quantile(pf_samples, 0.975)))
    return pf_samples, pf_mean, pf_cov, ci95


# ---------------------------------------------------------------------------
# EIER acquisition (NumPy)
# ---------------------------------------------------------------------------
class IntCache(NamedTuple):
    X_int: np.ndarray   # (m, d)
    muX: np.ndarray     # (m, 1), physical units
    varX: np.ndarray    # (m, 1), physical units
    stdX: np.ndarray    # (m, 1), physical units
    V: np.ndarray       # (m, budget)
    u_curr: float       # scalar H_n estimate over this integration batch


def build_int_cache_numpy(cache: GPCache, X_int: np.ndarray) -> IntCache:
    Kx = _matern52_cross(X_int, cache.X_train, cache.lengthscale, cache.kern_var)
    muX_norm = Kx.dot(cache.alpha)
    V = Kx.dot(cache.Kinv)
    varX_norm = cache.kern_var - np.sum(V * Kx, axis=1, keepdims=True)
    varX_norm = np.maximum(varX_norm, 1e-12)

    muX = cache.y_mean + cache.y_std * muX_norm
    varX = np.maximum((cache.y_std ** 2) * varX_norm, 1e-12)
    stdX = np.sqrt(varX)

    u = misclass_prob(muX, stdX)
    u_curr = float(np.mean(u))

    return IntCache(
        X_int=X_int,
        muX=muX,
        varX=varX,
        stdX=stdX,
        V=V,
        u_curr=u_curr,
    )


def eier_hnext_samples_batch_numpy(
    cache: GPCache,
    intc: IntCache,
    X_cand: np.ndarray,
    jitter_stddev: float,
    eps_z: np.ndarray,
    z_chunk_size: int = 64,
) -> np.ndarray:
    """Compute H_{n+1}(z_k) for one integration batch.

    Returns
    -------
    np.ndarray
        Shape (n_z, k) where k = number of candidates in X_cand.
    """
    eps = 1e-12
    jitter2 = float(jitter_stddev) ** 2

    Kp = _matern52_cross(X_cand, cache.X_train, cache.lengthscale, cache.kern_var)
    vp = Kp.dot(cache.Kinv)
    varp_norm = cache.kern_var - np.sum(vp * Kp, axis=1)
    varp_norm = np.maximum(varp_norm, eps)
    varp = (cache.y_std ** 2) * varp_norm
    # Correct GP conditioning denominator with observation jitter.
    denom = np.maximum(varp + jitter2, eps)

    k_Xp = _matern52_cross(intc.X_int, X_cand, cache.lengthscale, cache.kern_var)
    cov_norm = k_Xp - intc.V.dot(Kp.T)
    cov = (cache.y_std ** 2) * cov_norm

    varX_new = intc.varX - (cov ** 2) / denom[None, :]
    varX_new = np.maximum(varX_new, eps)
    stdX_new = np.sqrt(varX_new)
    gain = cov / denom[None, :]

    n_z = int(eps_z.shape[0])
    z_chunk = max(1, int(z_chunk_size))
    h_next_chunks = []
    sqrt_denom = np.sqrt(denom)[None, :]

    for z0 in range(0, n_z, z_chunk):
        z1 = min(n_z, z0 + z_chunk)
        eps_z_block = eps_z[z0:z1]  # (zb, k)
        delta_zk = sqrt_denom * eps_z_block

        # Shapes per z-block:
        # gain:      (m, k)
        # delta_zk:  (zb, k)
        # muX_new:   (zb, m, k)
        muX_new = intc.muX[:, 0][None, :, None] + gain[None, :, :] * delta_zk[:, None, :]
        tau_new = misclass_prob(muX_new, stdX_new[None, :, :])
        h_next_chunks.append(np.mean(tau_new, axis=1))  # (zb, k)

    return np.concatenate(h_next_chunks, axis=0)


def _merge_topk(scores_a: np.ndarray, x_a: np.ndarray, scores_b: np.ndarray, x_b: np.ndarray, k_keep: int):
    merged_scores = np.concatenate([scores_a, scores_b], axis=0)
    merged_x = np.concatenate([x_a, x_b], axis=0)
    idx = np.argpartition(merged_scores, -k_keep)[-k_keep:]
    idx = idx[np.argsort(merged_scores[idx])[::-1]]
    return merged_scores[idx], merged_x[idx]


def make_select_x_eier_shared_pool_numpy(
    dim: int,
    batch_size_acq: int,
    n_z_mc: int,
    jitter_stddev: float,
    local_mis_topk: int = 3000,
    debug_acq: bool = False,
):
    """Select x+ maximizing EIER on a fixed streamed pool S_pool."""
    bs_acq = int(batch_size_acq)
    n_z = int(n_z_mc)

    use_topk_filter = int(local_mis_topk) > 0
    k_keep = int(local_mis_topk) if use_topk_filter else bs_acq
    if use_topk_filter and k_keep <= 0:
        raise ValueError("local_mis_topk must be > 0 when filtering is enabled.")

    def select_x(
        cache: GPCache,
        X_pool_fixed: np.ndarray,
        z_seed: int,
    ):
        n_pool_total = int(X_pool_fixed.shape[0])
        n_pool_batches = (n_pool_total + bs_acq - 1) // bs_acq
        k_eff = min(k_keep, n_pool_total) if use_topk_filter else bs_acq

        if use_topk_filter:
            # Pass 1: shortlist candidates by local misclassification tau_n.
            top_scores = np.full((k_eff,), -np.inf, dtype=PRED_DTYPE)
            top_x = np.zeros((k_eff, dim), dtype=PRED_DTYPE)

            for b in range(n_pool_batches):
                start = b * bs_acq
                end = min(start + bs_acq, n_pool_total)
                X_cand = X_pool_fixed[start:end]
                mu, std = _predict_mu_std(cache, X_cand)
                local_score = misclass_prob(mu.reshape(-1), std.reshape(-1)).astype(PRED_DTYPE)
                top_scores, top_x = _merge_topk(top_scores, top_x, local_score, X_cand, k_eff)

            valid_short = np.isfinite(top_scores)
            eps_z_top = np.random.RandomState(int(z_seed)).normal(size=(n_z, k_eff)).astype(PRED_DTYPE)

            # Pass 2: evaluate EIER on shortlisted candidates, integrating H
            # over the full fixed pool S (paper protocol).
            h_curr_sum = 0.0
            h_next_sum = np.zeros((n_z, k_eff), dtype=PRED_DTYPE)
            done_pool = 0
            for b in range(n_pool_batches):
                start = b * bs_acq
                end = min(start + bs_acq, n_pool_total)
                X_int = X_pool_fixed[start:end]
                this_bs = X_int.shape[0]
                intc = build_int_cache_numpy(cache, X_int)
                h_curr_sum += intc.u_curr * float(this_bs)
                c0 = 0
                while c0 < k_eff:
                    c1 = min(k_eff, c0 + bs_acq)
                    h_next_zk = eier_hnext_samples_batch_numpy(
                        cache=cache,
                        intc=intc,
                        X_cand=top_x[c0:c1],
                        jitter_stddev=jitter_stddev,
                        eps_z=eps_z_top[:, c0:c1],
                    )
                    h_next_sum[:, c0:c1] += h_next_zk * float(this_bs)
                    c0 = c1
                done_pool += this_bs

            denom_total = max(float(done_pool), 1.0)
            h_curr = h_curr_sum / denom_total
            h_next = h_next_sum / denom_total
            expected_gain = np.mean(np.maximum(h_curr - h_next, 0.0), axis=0)
            expected_gain = np.where(valid_short, expected_gain, -np.inf)
            expected_gain = np.nan_to_num(expected_gain, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

            arg = int(np.argmax(expected_gain))
            best_x = top_x[arg]
            best_gain = float(max(expected_gain[arg], 0.0))

            if debug_acq:
                print(
                    f"[EIER dbg][topk] H_n={h_curr:.3e} | "
                    f"gain[min,max]=[{np.min(expected_gain):.3e},{np.max(expected_gain):.3e}] | "
                    f"valid_topk={int(np.sum(valid_short))}"
                )

            return best_x, best_gain, float(h_curr)

        # Fallback: evaluate all candidates.
        best_gain = -np.inf
        best_x = np.zeros((dim,), dtype=PRED_DTYPE)
        h_curr_ref = None
        rng_z = np.random.RandomState(int(z_seed))

        for b_cand in range(n_pool_batches):
            c_start = b_cand * bs_acq
            c_end = min(c_start + bs_acq, n_pool_total)
            X_cand = X_pool_fixed[c_start:c_end]
            this_cand_bs = X_cand.shape[0]
            eps_z = rng_z.normal(size=(n_z, this_cand_bs)).astype(PRED_DTYPE)

            h_curr_sum = 0.0
            h_next_sum = np.zeros((n_z, this_cand_bs), dtype=PRED_DTYPE)
            done_pool_int = 0
            for b_int in range(n_pool_batches):
                i_start = b_int * bs_acq
                i_end = min(i_start + bs_acq, n_pool_total)
                X_int = X_pool_fixed[i_start:i_end]
                this_int_bs = X_int.shape[0]
                intc = build_int_cache_numpy(cache, X_int)
                h_next_zk = eier_hnext_samples_batch_numpy(
                    cache=cache,
                    intc=intc,
                    X_cand=X_cand,
                    jitter_stddev=jitter_stddev,
                    eps_z=eps_z,
                )

                h_curr_sum += intc.u_curr * float(this_int_bs)
                h_next_sum += h_next_zk * float(this_int_bs)
                done_pool_int += this_int_bs

            denom_total = max(float(done_pool_int), 1.0)
            h_curr = h_curr_sum / denom_total
            if h_curr_ref is None:
                h_curr_ref = float(h_curr)
            h_next = h_next_sum / denom_total
            expected_gain = np.mean(np.maximum(h_curr - h_next, 0.0), axis=0)
            expected_gain = np.nan_to_num(expected_gain, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

            arg = int(np.argmax(expected_gain))
            bmax = float(expected_gain[arg])
            if bmax > best_gain:
                best_gain = bmax
                best_x = X_cand[arg]

        if h_curr_ref is None:
            h_curr_ref = 0.0
        return best_x, float(max(best_gain, 0.0)), h_curr_ref

    return select_x


# ---------------------------------------------------------------------------
# Argument parsing and configuration helpers
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GP-based Active Learning for structural reliability (EIER, NumPy)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="eier_default_np",
        help="Config name in config/ (e.g. 'eier_default_np') or explicit JSON path; use 'none' to disable.",
    )
    parser.add_argument("--casestudy", default="four_branch_7")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--budget", type=int, default=50, help="Total training points")
    parser.add_argument("--passive", type=int, default=10, help="Initial DoE size")
    parser.add_argument(
        "--obs-stddev",
        type=float,
        default=1e-6,
        help="Numerical jitter stddev (deterministic model; not learned noise).",
    )
    parser.add_argument("--n-mcs-pool", type=int, default=1_000_000)
    parser.add_argument(
        "--batch-size-acq",
        type=int,
        default=5_000,
        help="MC batch size for EIER candidate scan",
    )
    parser.add_argument(
        "--local-mis-topk",
        type=int,
        default=3_000,
        help="Evaluate EIER only on top-K candidates ranked by local misclassification; <=0 disables filtering.",
    )
    parser.add_argument(
        "--n-z-mc",
        type=int,
        default=64,
        help="Fantasy sample count for EIER look-ahead expectation over Z (also N_g for pf samples).",
    )
    parser.add_argument("--debug-acq", action="store_true", help="Enable acquisition debug prints.")
    return parser


def _resolve_config_path(config_arg: Optional[str]):
    if config_arg is None:
        return None

    cfg = config_arg.strip()
    if cfg.lower() in {"none", "null", ""}:
        return None

    p = Path(cfg)
    if p.suffix.lower() == ".json":
        if p.exists():
            return p
        alt = Path("config") / p.name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Config file not found: {cfg}")

    if p.exists():
        return p

    default_json = Path("config") / f"{cfg}.json"
    if default_json.exists():
        return default_json

    raise FileNotFoundError(
        f"Config '{cfg}' not found. Use a name in config/ or an explicit .json path."
    )


def _normalize_config_dict(cfg: dict, parser: argparse.ArgumentParser) -> dict:
    aliases = {
        "case_study": "casestudy",
        "total_samples": "budget",
        "passive_samples": "passive",
        "batch_size": "batch_size_acq",
        "topk_local_ibv": "local_mis_topk",
        "local_ibv_topk": "local_mis_topk",
        "n_z": "n_z_mc",
    }
    valid_keys = {a.dest for a in parser._actions}
    normalized = {}
    ignored = []

    for raw_key, value in cfg.items():
        key = raw_key.replace("-", "_")
        key = aliases.get(key, key)
        if key in valid_keys:
            normalized[key] = value
        else:
            ignored.append(raw_key)

    if ignored:
        print(f"[config] Ignoring unknown keys: {sorted(ignored)}")

    return normalized


def _parse_args() -> argparse.Namespace:
    parser = _build_parser()
    pre_args, _ = parser.parse_known_args()

    cfg_path = _resolve_config_path(pre_args.config)
    if cfg_path is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            parser.error(
                f"--config must contain a JSON object, got {type(cfg).__name__} in {cfg_path}."
            )
        parser.set_defaults(**_normalize_config_dict(cfg, parser))
        print(f"[config] Loaded: {cfg_path}")

    args = parser.parse_args()
    if args.seed is None:
        # If config uses `"seed": null`, draw one concrete seed for this run
        # so the printed config is fully reproducible.
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        print(f"[config] seed=null -> generated seed={args.seed}")

    return args


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    print("Experiment config:")
    print(json.dumps(vars(args), indent=2, sort_keys=True))

    lstate = ls_REGISTRY[args.casestudy]()
    Pf_ref = float(lstate.target_pf)
    dim = int(lstate.input_dim)

    n_train_final = int(args.budget)
    passive_samples = int(args.passive)
    obs_stddev = float(args.obs_stddev)
    n_mcs_pool = int(args.n_mcs_pool)
    batch_size_acq = int(args.batch_size_acq)
    local_mis_topk = int(args.local_mis_topk)
    n_z_mc = int(args.n_z_mc)
    debug_acq = bool(args.debug_acq)

    rng = np.random.RandomState(args.seed)

    x_train_norm, _, y_train_raw = lstate.get_doe(
        n_samples=passive_samples,
        method="lhs",
        random_state=rng,
    )
    x_train = np.asarray(x_train_norm, dtype=TRAIN_DTYPE)
    y_train = np.asarray(y_train_raw, dtype=TRAIN_DTYPE).reshape(-1, 1)

    n0 = int(x_train.shape[0])
    assert n0 <= n_train_final, f"n0={n0} exceeds budget={n_train_final}"

    Xbuf = np.zeros((n_train_final, dim), dtype=TRAIN_DTYPE)
    Ybuf = np.zeros((n_train_final, 1), dtype=TRAIN_DTYPE)
    Xbuf[:n0] = x_train
    Ybuf[:n0] = y_train

    n_train = n0
    iterations = n_train_final - n0

    print(f"Case={args.casestudy} | target Pf={Pf_ref:.6E}")
    print(f"Budget={n_train_final}. Initial DoE={n0} => iterations={iterations}.")
    print(f"Buffer DTYPE: {Xbuf.dtype}")
    print("Pf estimator: posterior GP trajectory samples on fixed S_pool")
    if local_mis_topk > 0:
        print(f"Acquisition filter: top-{local_mis_topk} candidates by local misclassification")
    else:
        print("Acquisition filter: disabled (evaluate EIER on all candidates)")
    print(f"EIER z-integration: n_z_mc={n_z_mc}")
    if local_mis_topk > n_mcs_pool:
        print(
            f"[warn] local_mis_topk={local_mis_topk} > n_mcs_pool={n_mcs_pool}. "
            f"Clamping to {n_mcs_pool}."
        )
        local_mis_topk = n_mcs_pool

    select_x_shared_pool = make_select_x_eier_shared_pool_numpy(
        dim=dim,
        batch_size_acq=batch_size_acq,
        n_z_mc=n_z_mc,
        jitter_stddev=obs_stddev,
        local_mis_topk=local_mis_topk,
        debug_acq=debug_acq,
    )

    # Single fixed pool S used by both EIER and pf posterior sampling.
    X_pool_fixed = rng.normal(size=(n_mcs_pool, dim)).astype(PRED_DTYPE)
    if debug_acq:
        print("[EIER dbg] X_pf_fixed[:2] = X_pool_fixed[:2] (same S)")
        print(X_pool_fixed[:2])
        print("[EIER dbg] X_pool_fixed[:2] =")
        print(X_pool_fixed[:2])

    timings = {"gp_train": [], "pf_estimate": [], "acquisition": [], "total": []}

    for it in range(iterations):
        iter_start = time.time()

        train_start = time.time()
        y_mean, y_std = _compute_y_scaler(Ybuf[:n_train])
        y_train_norm = (Ybuf[:n_train] - y_mean) / y_std
        obs_stddev_norm = float(obs_stddev / y_std)
        model = train_gp_numpy(
            X_train=Xbuf[:n_train],
            y_train_norm=y_train_norm,
            dim=dim,
            obs_stddev_norm=obs_stddev_norm,
        )
        cache = build_gp_cache_numpy(
            gpr=model,
            Xbuf=Xbuf,
            Ybuf=Ybuf,
            n_train=n_train,
            budget=n_train_final,
            obs_stddev_norm=obs_stddev_norm,
            y_mean=float(y_mean),
            y_std=float(y_std),
        )
        timings["gp_train"].append(time.time() - train_start)

        # z-seed changes per iteration for fantasy MC.
        seed_z = int(rng.randint(0, 2**31 - 1))

        # 2) Posterior trajectory pf samples on fixed S, using N_g = n_z_mc.
        pf_start = time.time()
        pf_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))
        _pf_samples, pf_mean, pf_cov, pf_ci95 = estimate_pf_posterior_samples(
            cache=cache,
            X_pool_fixed=X_pool_fixed,
            N_g=n_z_mc,
            batch_size_acq=batch_size_acq,
            rng=pf_rng,
        )
        timings["pf_estimate"].append(time.time() - pf_start)

        if np.isfinite(Pf_ref) and Pf_ref > 0.0:
            Pf_rel_diff = abs(pf_mean - Pf_ref) / Pf_ref
        else:
            Pf_rel_diff = np.nan

        # 4) EIER acquisition over filtered S_cand.
        acq_start = time.time()
        x_next_norm, eier_gain, h_curr_pool = select_x_shared_pool(
            cache,
            X_pool_fixed,
            seed_z,
        )
        timings["acquisition"].append(time.time() - acq_start)

        u_next_proxy = max(h_curr_pool - eier_gain, 0.0)
        stopping_triggered = False
        if debug_acq:
            assert x_next_norm.shape == (dim,), (
                f"Candidate shape mismatch: got {x_next_norm.shape}, expected {(dim,)}"
            )
            assert np.isfinite(eier_gain), "EIER gain is not finite."
            assert eier_gain >= -1e-10, f"EIER should be non-negative, got {eier_gain:.3e}"
            print(
                f"[EIER dbg][it={it:02d}] Pf_mean={pf_mean:.3E} | Pf_cov={pf_cov:.3E} | "
                f"H_n={h_curr_pool:.3E} | "
                f"best_x={np.asarray(x_next_norm)} | EIER={eier_gain:.3E}"
            )

        print(
            f"it = {it:02d} | n_train = {n_train:03d} | "
            f"pf_mean = {pf_mean:.3E} | pf_cov = {pf_cov:.3E} | "
            f"rel_diff = {Pf_rel_diff:.2E} | "
            f"H_n={h_curr_pool:.3E} | EIER={eier_gain:.3E} | U_next~={u_next_proxy:.3E} | "
            f"x_next={np.array2string(np.asarray(x_next_norm), precision=3)} | "
            f"stop={stopping_triggered} | "
            f"GP: {timings['gp_train'][-1]:.2f}s | "
            f"Pf: {timings['pf_estimate'][-1]:.2f}s | "
            f"Acq: {timings['acquisition'][-1]:.2f}s"
        )

        x_next_phys = isoprobabilistic_transform(
            np.asarray(x_next_norm, dtype=np.float64)[None, :],
            lstate.standard_marginals,
            lstate.physical_marginals,
        )
        y_next = lstate.eval_lstate(x_next_phys)

        Xbuf[n_train] = np.asarray(x_next_norm, dtype=TRAIN_DTYPE)
        Ybuf[n_train : n_train + 1] = np.asarray(y_next, dtype=TRAIN_DTYPE).reshape(1, 1)
        n_train += 1
        timings["total"].append(time.time() - iter_start)

    # Final model evaluation at full budget (no acquisition).
    final_start = time.time()
    train_start = time.time()
    y_mean, y_std = _compute_y_scaler(Ybuf[:n_train])
    y_train_norm = (Ybuf[:n_train] - y_mean) / y_std
    obs_stddev_norm = float(obs_stddev / y_std)
    model = train_gp_numpy(
        X_train=Xbuf[:n_train],
        y_train_norm=y_train_norm,
        dim=dim,
        obs_stddev_norm=obs_stddev_norm,
    )
    cache = build_gp_cache_numpy(
        gpr=model,
        Xbuf=Xbuf,
        Ybuf=Ybuf,
        n_train=n_train,
        budget=n_train_final,
        obs_stddev_norm=obs_stddev_norm,
        y_mean=float(y_mean),
        y_std=float(y_std),
    )
    timings["gp_train"].append(time.time() - train_start)

    pf_start = time.time()
    pf_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))
    _pf_samples, pf_mean, pf_cov, _pf_ci95 = estimate_pf_posterior_samples(
        cache=cache,
        X_pool_fixed=X_pool_fixed,
        N_g=n_z_mc,
        batch_size_acq=batch_size_acq,
        rng=pf_rng,
    )
    timings["pf_estimate"].append(time.time() - pf_start)

    if np.isfinite(Pf_ref) and Pf_ref > 0.0:
        Pf_rel_diff = abs(pf_mean - Pf_ref) / Pf_ref
    else:
        Pf_rel_diff = np.nan
    timings["total"].append(time.time() - final_start)

    print(
        f"it = {iterations:02d} | n_train = {n_train:03d} | "
        f"pf_mean = {pf_mean:.3E} | pf_cov = {pf_cov:.3E} | rel_diff = {Pf_rel_diff:.2E} | "
        f"FINAL evaluation (no acquisition)"
    )

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    total_wall = np.sum(timings["total"]) if timings["total"] else 0.0
    for key_name, times in timings.items():
        if times:
            avg_time = float(np.mean(times))
            total_time = float(np.sum(times))
            pct = (total_time / total_wall) * 100.0 if total_wall > 0 else 0.0
            print(
                f"{key_name:15s}: avg={avg_time:6.3f}s | "
                f"total={total_time:7.2f}s | {pct:5.1f}%"
            )
    print("=" * 80)
    print(f"Final pf_mean = {pf_mean:.3E} | pf_cov = {pf_cov:.3E} | rel_diff = {Pf_rel_diff:.2E} |")


if __name__ == "__main__":
    main()
