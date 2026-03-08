import os
import re
from contextlib import nullcontext

import numpy as np
import scipy.stats as stats
from scipy.optimize import fmin_l_bfgs_b
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

def isoprobabilistic_transform(x, source_marginals, target_marginals):
    # Ensure x is a numpy array
    x = np.atleast_2d(x).astype(np.float64)
    transformed_x = np.empty_like(x)

    for i, (source_params, target_params) in enumerate(zip(source_marginals.values(), target_marginals.values())):
        loc_s, scale_s, dist_s_name = source_params
        loc_t, scale_t, dist_t_name = target_params

        # Define source distribution
        if dist_s_name == 'lognorm':
            mu_s = np.log(loc_s**2 / np.sqrt(loc_s**2 + scale_s**2))
            sigma_s = np.sqrt(np.log(1 + (scale_s / loc_s)**2))
            dist_source = stats.lognorm(s=sigma_s, scale=np.exp(mu_s))
        elif dist_s_name == 'uniform':
            dist_source = stats.uniform(loc=loc_s, scale=scale_s)
        else:
            dist_source = getattr(stats, dist_s_name)(loc=loc_s, scale=scale_s)

        # Define target distribution
        if dist_t_name == 'lognorm':
            mu_t = np.log(loc_t**2 / np.sqrt(loc_t**2 + scale_t**2))
            sigma_t = np.sqrt(np.log(1 + (scale_t / loc_t)**2))
            dist_target = stats.lognorm(s=sigma_t, scale=np.exp(mu_t))
        elif dist_t_name == 'uniform':
            # Correct the scale for the uniform distribution
            dist_target = stats.uniform(loc=loc_t, scale=scale_t - loc_t)
        else:
            dist_target = getattr(stats, dist_t_name)(loc=loc_t, scale=scale_t)

        # Compute transformation: Target_PPF(Source_CDF(x))
        cdf_source = dist_source.cdf(x[:, i])
        transformed_x[:, i] = dist_target.ppf(cdf_source)

    return transformed_x.squeeze() if x.shape[0] == 1 else transformed_x

def custom_optimizer(obj_func, initial_theta, bounds):
    opt_res = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=1000)
    return opt_res[0], opt_res[1]

# Function to make predictions over a batch of samples
def predict_batch(model, x_batch):
    return model.predict(x_batch, return_std=True)


def _parse_positive_int(raw_value):
    if raw_value is None:
        return None
    match = re.search(r"\d+", str(raw_value))
    if match is None:
        return None
    value = int(match.group())
    return value if value > 0 else None


def resolve_cpu_workers(value):
    value = int(value)

    slurm_cpus_per_task = _parse_positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    try:
        affinity_count = max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        count = os.cpu_count()
        affinity_count = 1 if count is None else max(1, int(count))

    available_workers = affinity_count
    if slurm_cpus_per_task is not None:
        available_workers = min(available_workers, slurm_cpus_per_task)

    if value == -1:
        return int(available_workers)
    if value < 1:
        raise ValueError("`cpu_workers` must be a positive integer or -1.")
    if value > available_workers:
        print(
            f"[cpu] Requested {value} workers but only {available_workers} are available "
            "for this task. Capping worker count."
        )
    return int(min(value, available_workers))


def _resolve_n_jobs(n_jobs, n_batches):
    n_jobs = int(n_jobs)
    if n_jobs == -1:
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except AttributeError:
            affinity_count = os.cpu_count() or 1
        n_jobs = max(1, int(affinity_count))
    elif n_jobs < 1:
        raise ValueError("`n_jobs` must be a positive integer or -1.")

    slurm_cpus_per_task = _parse_positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if slurm_cpus_per_task is not None:
        n_jobs = min(n_jobs, slurm_cpus_per_task)

    return max(1, min(int(n_jobs), int(n_batches)))


def parallel_predict(model_gp, x_mc_pool, n_jobs=-1, batch_size=10000, prefer="threads"):
    x_mc_pool = np.asarray(x_mc_pool, dtype=np.float64)
    batch_size = max(1, int(batch_size))
    n_batches = int(np.ceil(x_mc_pool.shape[0] / batch_size))
    n_jobs = _resolve_n_jobs(n_jobs, n_batches)

    # Split into batches
    batches = [x_mc_pool[i * batch_size: (i + 1) * batch_size] for i in range(n_batches)]

    if n_batches == 1 or int(n_jobs) == 1:
        results = [predict_batch(model_gp, batch) for batch in batches]
    else:
        limit_ctx = threadpool_limits(limits=1, user_api="blas") if threadpool_limits is not None else nullcontext()
        try:
            with limit_ctx:
                results = Parallel(n_jobs=n_jobs, prefer=prefer)(
                    delayed(predict_batch)(model_gp, batch) for batch in batches
                )
        except RuntimeError as exc:
            if "can't start new thread" not in str(exc).lower():
                raise
            print(
                f"[parallel_predict] Thread creation failed with n_jobs={n_jobs}; "
                "falling back to sequential execution for this call."
            )
            results = [predict_batch(model_gp, batch) for batch in batches]

    # Combining results
    means, stds = zip(*results)
    mean_prediction = np.concatenate(means, axis=0).astype(np.float64, copy=False)
    std_prediction = np.concatenate(stds, axis=0).astype(np.float64, copy=False)
    return mean_prediction, std_prediction

def normalize_array(arr):
    arr = np.asarray(arr, dtype=np.float64)
    min_vals = np.min(arr, axis=0, keepdims=True)
    max_vals = np.max(arr, axis=0, keepdims=True)
    denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
    normalized = (arr - min_vals) / denom
    return normalized.squeeze() if arr.ndim == 1 else normalized
