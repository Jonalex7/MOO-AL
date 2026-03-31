import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

class g2D_hat_function():
    def __init__(self):
        self.input_dim = 2
        self.output_dim = 1
        self.target_pf = 0.0003803
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim)}
        self.physical_marginals = {'x1': [0.25, 1.0, 'norm'],
                                   'x2': [0.25, 1.0, 'norm']}
        '''mean(or min), std(or max), marginal_distrib'''

    def eval_lstate(self, x):
        msg = 'Ok'
        x = np.atleast_2d(x).astype(np.float64)

        if x.ndim > 2:
            msg = 'Only available for 1D and 2D arrays.'
            return np.nan, msg

        nrv_p = x.shape[1]
        if nrv_p != self.input_dim:
            msg = f'Expected {self.input_dim} vars, got {nrv_p}!'
            return np.nan, msg

        g_val_sys = 20 - (x[:, 0] - x[:, 1]) ** 2 - 8 * (x[:, 0] + x[:, 1] - 4) ** 3
        return g_val_sys
    
    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.normal(0, 1, size=(n_mcs, self.input_dim))
        x_mc_physical = isoprobabilistic_transform(x_mc_norm, self.standard_marginals, self.physical_marginals)
        y_mc = self.eval_lstate(x_mc_physical)
        Pf_ref = np.mean(y_mc < 0)
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref, B_ref, x_mc_physical, y_mc

    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.default_rng(random_state)
        
        # Generates samples that are uniformly distributed within the unit hypercube [0,1]^d
        uniform_marginals = {f'x{var+1}': [0, 1.0, 'uniform'] for var in range(self.input_dim)}
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_uniform = sampler.random(n=int(n_samples))

        # Converting samples from uniform to physical and standard space
        x_doe_physical = isoprobabilistic_transform(x_uniform, uniform_marginals, self.physical_marginals)
        x_doe_norm = isoprobabilistic_transform(x_uniform, uniform_marginals, self.standard_marginals)
        y_scaled = self.eval_lstate(x_doe_physical)

        return x_doe_norm, x_doe_physical, y_scaled
