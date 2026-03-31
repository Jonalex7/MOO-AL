import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

NAME = "wei_g3"
DIM = 3


class g3d_wei_g3():
    """Wei et al. (Eq. 20c) benchmark limit-state function.

    g3(x1, x2, x3) =
      17.1 - 20*exp(-0.2*sqrt(0.2*sum_i x_i^2))
      - exp(0.2*sum_i cos(2*pi*x_i)) + exp(1)
    """

    def __init__(self):
        self.input_dim = DIM
        self.output_dim = 1
        self.target_pf = 0.07806069
        self.standard_marginals = {f"x{var+1}": [0, 1.0, "norm"] for var in range(self.input_dim)}
        self.physical_marginals = {f"x{var+1}": [0, 1.0, "norm"] for var in range(self.input_dim)}
        """mean(or min), std(or max), marginal_distrib"""

    def eval_lstate(self, x):
        msg = "Ok"
        x = np.atleast_2d(x).astype(np.float64)

        if x.ndim > 2:
            msg = "Only available for 1D and 2D arrays."
            return np.nan, msg

        nrv_p = x.shape[1]
        if nrv_p != self.input_dim:
            msg = f"Expected {self.input_dim} vars, got {nrv_p}!"
            return np.nan, msg

        sum_sq = np.sum(x[:, :3] ** 2, axis=1)
        sum_cos = np.sum(np.cos(2.0 * np.pi * x[:, :3]), axis=1)

        g_val_sys = (
            17.1
            - 20.0 * np.exp(-0.2 * np.sqrt(0.2 * sum_sq))
            - np.exp(0.2 * sum_cos)
            + np.exp(1.0)
        )
        return g_val_sys

    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.normal(0, 1, size=(n_mcs, self.input_dim))
        x_mc_physical = isoprobabilistic_transform(
            x_mc_norm, self.standard_marginals, self.physical_marginals
        )
        y_mc = self.eval_lstate(x_mc_physical)
        Pf_ref = np.mean(y_mc < 0)
        B_ref = -norm.ppf(Pf_ref)
        return Pf_ref, B_ref, x_mc_physical, y_mc

    def get_doe(self, n_samples=10, method="lhs", random_state=None):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.default_rng(random_state)

        uniform_marginals = {f"x{var+1}": [0, 1.0, "uniform"] for var in range(self.input_dim)}
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_uniform = sampler.random(n=int(n_samples))

        x_doe_physical = isoprobabilistic_transform(
            x_uniform, uniform_marginals, self.physical_marginals
        )
        x_doe_norm = isoprobabilistic_transform(
            x_uniform, uniform_marginals, self.standard_marginals
        )
        y_scaled = self.eval_lstate(x_doe_physical)

        return x_doe_norm, x_doe_physical, y_scaled
