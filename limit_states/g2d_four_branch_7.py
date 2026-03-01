import numpy as np
from scipy.stats import norm, qmc
from utils.data import isoprobabilistic_transform 

class g2D_four_branch_7():
    def __init__(self):
        self.input_dim = 2
        self.output_dim = 1
        self.target_pf = 0.002265 
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim)}
        self.physical_marginals = {'x1': [0, 1.0, 'norm'],
                                   'x2': [0, 1.0, 'norm']}

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
        
        # The four limit state components
        g1 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 - (x[:, 0] + x[:, 1])/np.sqrt(2)
        g2 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 + (x[:, 0] + x[:, 1])/np.sqrt(2)
        g3 = (x[:, 0] - x[:, 1]) + (7/np.sqrt(2))
        g4 = (x[:, 1] - x[:, 0]) + (7/np.sqrt(2))
        
        # System failure occurs if any component fails (min of the components)
        g_val_sys = np.amin(np.stack((g1, g2, g3, g4)), axis=0)

        return g_val_sys

    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        # Generate samples in standard normal space
        x_mc_norm = np.random.normal(0, 1, size=(n_mcs, self.input_dim))
        
        # Transform to physical space
        x_mc_physical = isoprobabilistic_transform(x_mc_norm, self.standard_marginals, self.physical_marginals)
        
        # Evaluate performance function
        y_mc = self.eval_lstate(x_mc_physical)
        
        # Probability of failure: Pf = P(G <= 0)
        # In NumPy, (y_mc < 0) creates a boolean mask; .mean() treats True as 1 and False as 0
        Pf_ref = np.mean(y_mc < 0)
        
        # Reliability index Beta = -Phi^-1(Pf)
        B_ref = -norm.ppf(Pf_ref)
        
        return Pf_ref, B_ref, x_mc_physical, y_mc
    
    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        # Handle random state for reproducibility
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.default_rng(random_state)
        
        uniform_marginals = {f'x{var+1}': [0, 1.0, 'uniform'] for var in range(self.input_dim)}
        
        # Use Scipy's QMC for Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_uniform = sampler.random(n=int(n_samples))

        x_doe_physical = isoprobabilistic_transform(x_uniform, uniform_marginals, self.physical_marginals)
        x_doe_norm = isoprobabilistic_transform(x_uniform, uniform_marginals, self.standard_marginals)
        y_scaled = self.eval_lstate(x_doe_physical)

        return x_doe_norm, x_doe_physical, y_scaled