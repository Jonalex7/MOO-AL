import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

NAME = "wei_g1"
DIM = 2


class g2d_wei_g1():
    """Wei et al. (Eq. 20a) benchmark limit-state function.

    g1(x1, x2) = sum_{i=1..4} c_i * exp(-a_{i1}(x1-beta_{i1})^2 - a_{i2}(x2-beta_{i2})^2)
    """

    def __init__(self):
        self.input_dim = DIM
        self.output_dim = 1
        self.target_pf = np.nan
        self.standard_marginals = {f"x{var+1}": [0, 1.0, "norm"] for var in range(self.input_dim)}
        self.physical_marginals = {f"x{var+1}": [0, 1.0, "norm"] for var in range(self.input_dim)}
        """mean(or min), std(or max), marginal_distrib"""

    def eval_lstate(self, x):
        msg = "Ok"
        x = np.atleast_2d(x).astype(np.float64)

        # Only allow 2D arrays (N, d)
        if x.ndim != 2:
            msg = "Only available for 1D and 2D arrays."
            return np.nan, msg

        # This limit-state is defined only for d=2
        if x.shape[1] != 2:
            msg = f"g1 is defined for 2 variables (x1,x2). Got shape {x.shape}."
            return np.nan, msg

        # Parameters from the paper: (2x4)^T -> (4x2)
        # Each row i is [alpha_i1, alpha_i2]
        alpha = np.array([
            [2.0, 3.0],  # i=1: alpha_11=2, alpha_12=3
            [3.0, 2.0],  # i=2: alpha_21=3, alpha_22=2
            [1.0, 4.0],  # i=3: alpha_31=1, alpha_32=4
            [4.0, 1.0],  # i=4: alpha_41=4, alpha_42=1
        ], dtype=np.float64)  # (4,2)

        # Each row i is [beta_i1, beta_i2]
        beta = np.array([
            [-0.5, -0.5],  # i=1
            [ 0.5, -0.5],  # i=2
            [-0.5,  0.5],  # i=3
            [ 0.5,  0.5],  # i=4
        ], dtype=np.float64)  # (4,2)

        # c has exactly 4 elements
        c = np.array([1.0, -1.5, -1.5, 2.0], dtype=np.float64)  # (4,)

        # Broadcasted differences:
        # x1: (N,1), beta[:,0]: (4,) -> dx1: (N,4)
        # x2: (N,1), beta[:,1]: (4,) -> dx2: (N,4)
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        dx1 = x1 - beta[None, :, 0]
        dx2 = x2 - beta[None, :, 1]

        exponent = -(alpha[None, :, 0] * dx1**2 + alpha[None, :, 1] * dx2**2)  # (N,4)
        g_vals = np.sum(c[None, :] * np.exp(exponent), axis=1)  # (N,)

        return g_vals

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
