from typing import NamedTuple

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


PRED_DTYPE = np.float64

class GPCache(NamedTuple):
    X_train: np.ndarray
    alpha: np.ndarray
    Kinv: np.ndarray
    lengthscale: np.ndarray
    kern_var: float
    y_mean: float
    y_std: float
    n_train: int


class IntCache(NamedTuple):
    X_int: np.ndarray
    muX: np.ndarray
    varX: np.ndarray
    stdX: np.ndarray
    V: np.ndarray
    u_curr: float


def _matern52_cross(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: np.ndarray,
    variance: float,
) -> np.ndarray:
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


def build_gp_cache_from_gpr(gpr: GaussianProcessRegressor) -> GPCache:
    kernel_ = gpr.kernel_
    if not (hasattr(kernel_, "k1") and hasattr(kernel_, "k2")):
        raise RuntimeError(f"Unexpected fitted kernel format: {kernel_!r}")

    kern_var = float(kernel_.k1.constant_value)
    lengthscale = np.asarray(kernel_.k2.length_scale, dtype=PRED_DTYPE)
    X_train = np.asarray(gpr.X_train_, dtype=PRED_DTYPE)
    n_train = int(X_train.shape[0])

    alpha = np.asarray(gpr.alpha_, dtype=PRED_DTYPE).reshape(-1, 1)
    eye = np.eye(n_train, dtype=PRED_DTYPE)
    Kinv = np.linalg.solve(gpr.L_.T, np.linalg.solve(gpr.L_, eye))

    y_mean = getattr(gpr, "_y_train_mean", 0.0)
    y_std = getattr(gpr, "_y_train_std", 1.0)
    y_mean = float(np.asarray(y_mean, dtype=PRED_DTYPE).reshape(-1)[0])
    y_std = float(np.asarray(y_std, dtype=PRED_DTYPE).reshape(-1)[0])
    if not np.isfinite(y_std) or y_std <= 0.0:
        y_std = 1.0

    return GPCache(
        X_train=X_train,
        alpha=alpha,
        Kinv=Kinv.astype(PRED_DTYPE),
        lengthscale=lengthscale,
        kern_var=np.float64(kern_var),
        y_mean=np.float64(y_mean),
        y_std=np.float64(y_std),
        n_train=n_train,
    )


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
    z = -np.abs(mu) / np.maximum(std, 1e-12)
    return norm.cdf(z)


def _posterior_cov_cross(cache: GPCache, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    K_ab = _matern52_cross(XA, XB, cache.lengthscale, cache.kern_var)
    K_ax = _matern52_cross(XA, cache.X_train, cache.lengthscale, cache.kern_var)
    K_bx = _matern52_cross(XB, cache.X_train, cache.lengthscale, cache.kern_var)
    return (K_ab - (K_ax.dot(cache.Kinv)).dot(K_bx.T)).astype(PRED_DTYPE)


def estimate_pf_posterior_samples(
    cache: GPCache,
    X_pool_fixed: np.ndarray,
    N_g: int,
    batch_size_acq: int,
    rng: np.random.RandomState,
):
    X_pool_fixed = np.asarray(X_pool_fixed, dtype=PRED_DTYPE)
    N = int(X_pool_fixed.shape[0])
    N_g = int(N_g)
    bs = int(batch_size_acq)
    if N <= 0 or N_g <= 0:
        raise ValueError("X_pool_fixed and N_g must be positive.")

    # Low-rank posterior trajectory approximation on the fixed pool.
    M = min(bs, N)
    idx_ind = rng.choice(N, size=M, replace=False)
    X_ind = X_pool_fixed[idx_ind]

    mu_pool = np.zeros((N,), dtype=PRED_DTYPE)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        mu_b, _ = _predict_mu_std(cache, X_pool_fixed[s:e])
        mu_pool[s:e] = mu_b.reshape(-1)

    C_SI = np.zeros((N, M), dtype=PRED_DTYPE)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        C_SI[s:e] = _posterior_cov_cross(cache, X_pool_fixed[s:e], X_ind)

    C_II = _posterior_cov_cross(cache, X_ind, X_ind)
    C_II = 0.5 * (C_II + C_II.T)

    jitter = np.asarray(1e-10, dtype=PRED_DTYPE)
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

    B = np.linalg.solve(L_II, C_SI.T).T

    pf_samples = np.zeros((N_g,), dtype=PRED_DTYPE)
    g_chunk = max(1, min(N_g, bs))
    for g0 in range(0, N_g, g_chunk):
        g1 = min(N_g, g0 + g_chunk)
        eps = rng.normal(size=(M, g1 - g0)).astype(PRED_DTYPE)
        g_draws = mu_pool[:, None] + B.dot(eps)
        pf_samples[g0:g1] = np.mean(g_draws < 0.0, axis=0)

    pf_mean = float(np.mean(pf_samples))
    pf_std = float(np.std(pf_samples, ddof=1)) if N_g > 1 else 0.0
    pf_cov = float(pf_std / max(pf_mean, 1e-16))
    ci95 = (float(np.quantile(pf_samples, 0.025)), float(np.quantile(pf_samples, 0.975)))
    return pf_samples, pf_mean, pf_cov, ci95


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
    eps = 1e-12
    jitter2 = float(jitter_stddev) ** 2

    Kp = _matern52_cross(X_cand, cache.X_train, cache.lengthscale, cache.kern_var)
    vp = Kp.dot(cache.Kinv)
    varp_norm = cache.kern_var - np.sum(vp * Kp, axis=1)
    varp_norm = np.maximum(varp_norm, eps)
    varp = (cache.y_std ** 2) * varp_norm
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
        eps_z_block = eps_z[z0:z1]
        delta_zk = sqrt_denom * eps_z_block

        muX_new = intc.muX[:, 0][None, :, None] + gain[None, :, :] * delta_zk[:, None, :]
        tau_new = misclass_prob(muX_new, stdX_new[None, :, :])
        h_next_chunks.append(np.mean(tau_new, axis=1))

    return np.concatenate(h_next_chunks, axis=0)


def _merge_topk_indices(
    scores_a: np.ndarray,
    idx_a: np.ndarray,
    scores_b: np.ndarray,
    idx_b: np.ndarray,
    k_keep: int,
):
    merged_scores = np.concatenate([scores_a, scores_b], axis=0)
    merged_idx = np.concatenate([idx_a, idx_b], axis=0)
    take = min(int(k_keep), merged_scores.shape[0])
    part = np.argpartition(merged_scores, merged_scores.shape[0] - take)[-take:]
    part = part[np.argsort(merged_scores[part])[::-1]]
    return merged_scores[part], merged_idx[part]


def select_eier_index(
    model_gp: GaussianProcessRegressor,
    candidate_pool: np.ndarray,
    batch_size_acq: int,
    n_z_mc: int,
    jitter_stddev: float,
    local_mis_topk: int = 3000,
    z_seed: int = 0,
    debug_acq: bool = False,
    skip_indices=None,
) -> int:
    candidate_pool = np.asarray(candidate_pool, dtype=PRED_DTYPE)
    if candidate_pool.ndim != 2 or candidate_pool.shape[0] == 0:
        raise ValueError("`candidate_pool` must be a non-empty 2D array.")

    cache = build_gp_cache_from_gpr(model_gp)
    n_pool_total, dim = candidate_pool.shape
    bs_acq = int(batch_size_acq)
    n_z = int(n_z_mc)
    candidate_mask = np.ones((n_pool_total,), dtype=bool)
    if skip_indices is not None:
        skip_idx = np.asarray(skip_indices, dtype=int).reshape(-1)
        if skip_idx.size > 0:
            valid_skip = skip_idx[(skip_idx >= 0) & (skip_idx < n_pool_total)]
            candidate_mask[valid_skip] = False
    if not np.any(candidate_mask):
        raise RuntimeError("EIER candidate set is empty after applying skip_indices.")

    use_topk_filter = int(local_mis_topk) > 0
    k_keep = min(int(local_mis_topk), n_pool_total) if use_topk_filter else bs_acq
    n_pool_batches = (n_pool_total + bs_acq - 1) // bs_acq

    if use_topk_filter:
        top_scores = np.full((k_keep,), -np.inf, dtype=PRED_DTYPE)
        top_idx = np.zeros((k_keep,), dtype=int)

        for b in range(n_pool_batches):
            start = b * bs_acq
            end = min(start + bs_acq, n_pool_total)
            X_cand = candidate_pool[start:end]
            mu, std = _predict_mu_std(cache, X_cand)
            local_score = misclass_prob(mu.reshape(-1), std.reshape(-1)).astype(PRED_DTYPE)
            local_score = np.where(candidate_mask[start:end], local_score, -np.inf)
            batch_idx = np.arange(start, end, dtype=int)
            top_scores, top_idx = _merge_topk_indices(top_scores, top_idx, local_score, batch_idx, k_keep)

        valid_short = np.isfinite(top_scores)
        short_idx = top_idx[valid_short]
        if short_idx.size == 0:
            raise RuntimeError("EIER shortlist is empty.")

        eps_z_top = np.random.RandomState(int(z_seed)).normal(size=(n_z, short_idx.size)).astype(PRED_DTYPE)
        h_curr_sum = 0.0
        h_next_sum = np.zeros((n_z, short_idx.size), dtype=PRED_DTYPE)
        done_pool = 0

        for b in range(n_pool_batches):
            start = b * bs_acq
            end = min(start + bs_acq, n_pool_total)
            X_int = candidate_pool[start:end]
            this_bs = X_int.shape[0]
            intc = build_int_cache_numpy(cache, X_int)
            h_curr_sum += intc.u_curr * float(this_bs)

            c0 = 0
            while c0 < short_idx.size:
                c1 = min(short_idx.size, c0 + bs_acq)
                h_next_zk = eier_hnext_samples_batch_numpy(
                    cache=cache,
                    intc=intc,
                    X_cand=candidate_pool[short_idx[c0:c1]],
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
        expected_gain = np.nan_to_num(expected_gain, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

        if debug_acq:
            print(
                f"[EIER dbg][topk] H_n={h_curr:.3e} | "
                f"gain[min,max]=[{np.min(expected_gain):.3e},{np.max(expected_gain):.3e}] | "
                f"valid_topk={int(short_idx.size)}"
            )

        return int(short_idx[int(np.argmax(expected_gain))])

    best_gain = -np.inf
    best_idx = 0
    rng_z = np.random.RandomState(int(z_seed))

    for b_cand in range(n_pool_batches):
        c_start = b_cand * bs_acq
        c_end = min(c_start + bs_acq, n_pool_total)
        cand_abs_idx = np.arange(c_start, c_end, dtype=int)
        cand_abs_idx = cand_abs_idx[candidate_mask[c_start:c_end]]
        if cand_abs_idx.size == 0:
            continue
        X_cand = candidate_pool[cand_abs_idx]
        this_cand_bs = X_cand.shape[0]
        eps_z = rng_z.normal(size=(n_z, this_cand_bs)).astype(PRED_DTYPE)

        h_curr_sum = 0.0
        h_next_sum = np.zeros((n_z, this_cand_bs), dtype=PRED_DTYPE)
        done_pool_int = 0
        for b_int in range(n_pool_batches):
            i_start = b_int * bs_acq
            i_end = min(i_start + bs_acq, n_pool_total)
            X_int = candidate_pool[i_start:i_end]
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
        h_next = h_next_sum / denom_total
        expected_gain = np.mean(np.maximum(h_curr - h_next, 0.0), axis=0)
        expected_gain = np.nan_to_num(expected_gain, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

        arg = int(np.argmax(expected_gain))
        gain = float(expected_gain[arg])
        if gain > best_gain:
            best_gain = gain
            best_idx = int(cand_abs_idx[arg])

    return int(best_idx)
