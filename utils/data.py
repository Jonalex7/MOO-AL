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

def parallel_predict(model_gp, x_mc_pool, n_jobs=-1, batch_size=10000, prefer="threads"):
    x_mc_pool = np.asarray(x_mc_pool, dtype=np.float64)
    batch_size = max(1, int(batch_size))
    n_batches = int(np.ceil(x_mc_pool.shape[0] / batch_size))

    # Split into batches
    batches = [x_mc_pool[i * batch_size: (i + 1) * batch_size] for i in range(n_batches)]

    if n_batches == 1 or int(n_jobs) == 1:
        results = [predict_batch(model_gp, batch) for batch in batches]
    else:
        limit_ctx = threadpool_limits(limits=1, user_api="blas") if threadpool_limits is not None else nullcontext()
        with limit_ctx:
            results = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(predict_batch)(model_gp, batch) for batch in batches
            )

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
