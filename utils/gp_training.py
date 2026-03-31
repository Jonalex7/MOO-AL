import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from utils.data import custom_optimizer


def make_base_kernel(input_dim, upper_bound=1e5):
    upper_bound = float(upper_bound)
    length_init = np.full(input_dim, 1.0, dtype=np.float64)
    kernel = ConstantKernel(1.0, (1e-5, upper_bound)) * Matern(
        length_scale=length_init,
        length_scale_bounds=(1e-5, upper_bound),
        nu=2.5,
    )
    return kernel


def is_bad_fit(current_lml, prev_lml, lml_drop_tol=50.0, abs_lml_low=-100.0):
    """
    Consider a fit 'bad' if:
      - LML drops a lot compared to the previous good model, OR
      - LML is absolutely very low.
    """
    too_low = current_lml < abs_lml_low
    if prev_lml is None:
        big_drop = False
    else:
        big_drop = current_lml < (prev_lml - lml_drop_tol)
    return too_low or big_drop


def resolve_gp_alpha(config, y_train):
    y_scale = max(float(np.std(y_train)), 1e-12)
    alpha = float(config["obs_stddev"] / y_scale) ** 2
    return max(alpha, float(config.get("min_gp_alpha", 1e-12)))


def fit_gp_with_optional_stabilization(
    x_train_norm,
    y_train,
    init_kernel,
    input_dim,
    gp_alpha,
    n_restarts_optimizer,
    enable_stabilization=False,
    max_gp_alpha=1e-4,
):
    def _build_model(kernel, alpha):
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True,
            optimizer=custom_optimizer,
            alpha=float(alpha),
        )

    model_gp = _build_model(init_kernel, gp_alpha)
    try:
        model_gp.fit(x_train_norm, y_train)
        return model_gp, {
            "stabilized": False,
            "alpha_used": float(gp_alpha),
            "retry_count": 0,
            "kernel_upper": 1e5,
        }
    except np.linalg.LinAlgError:
        if not enable_stabilization:
            raise

    retry_count = 0
    gp_alpha_retry = max(float(gp_alpha), 1e-12)
    while gp_alpha_retry <= float(max_gp_alpha) + 1e-18:
        for kernel_upper in (1e5, 30.0):
            retry_count += 1
            retry_kernel = make_base_kernel(input_dim, upper_bound=kernel_upper)
            retry_model = _build_model(retry_kernel, gp_alpha_retry)
            try:
                retry_model.fit(x_train_norm, y_train)
                return retry_model, {
                    "stabilized": True,
                    "alpha_used": float(gp_alpha_retry),
                    "retry_count": int(retry_count),
                    "kernel_upper": float(kernel_upper),
                }
            except np.linalg.LinAlgError:
                continue
        gp_alpha_retry *= 10.0

    raise np.linalg.LinAlgError(
        "GP fit failed after stabilization retries (alpha backoff + tighter kernel bounds)."
    )
