import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

'''Example 4: The Himmelblau Function Ref. (HMCMC High Dim, Prof. Kostas, pag. 16) 
which is particularly suitable for reliability examples with multiple separated failure domains. 
x1 and x2 are assumed to be independent standard normal random variables and 
the constant beta is used to define different levels of the failure probability. 
(beta = 95 for Ref. PF=1.65E-4)   (beta = 50 for Ref. PF=2.77E-7) (beta = 65 for Ref. PF=2.8713e-6) '''

class g2d_himmelblau():
    def __init__(self):
        self.input_dim = 2
        self.output_dim = 1
        self.target_pf = 0.0001674 # ref with MCS = 1e7
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim )}

        self.physical_marginals = {'x1': [0, 1.0, 'norm'],
                          'x2': [0, 1.0, 'norm']}
        '''mean(or min), std(or max), marginal_distrib'''

    def eval_lstate(self, x):
            # x = np.array(x, dtype='f') # <-- REMOVE or change to 'd' (float64)
            # Remove dtype='f'. Use default np.array(x) which is typically float64.
            x = np.array(x) 
            
            n_dim = len(x.shape)
            if n_dim == 1:
                x = np.array(x)[np.newaxis]
                
            beta = 95
            # The calculation will now use the float64 precision of the input array x.
            term1 = (((0.75*x[:,0] - 0.5)**2 / 1.81) + ((0.75*x[:,1] - 0.5) / 1.81) - 11)**2
            term2 = (((0.75*x[:,0] - 1.0)/ 1.81) + ((0.75*x[:,1] - 0.5)**2 / 1.81) - 7)**2
            g = term1 + term2 - beta
            
            # Explicitly set dtype=torch.float64
            return torch.tensor(g, dtype=torch.float64)    

    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.normal(0, 1, size=(n_mcs, self.input_dim))

        x_mc_physical = isoprobabilistic_transform(x_mc_norm, self.standard_marginals, self.physical_marginals)
        y_mc = self.eval_lstate(x_mc_physical)
        Pf_ref = (y_mc < 0).double().mean()
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref.item(), B_ref, x_mc_physical, y_mc

    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()

        if method == 'lhs':
            uniform_marginals = {f'x{var+1}': [0, 1.0, 'uniform'] for var in range(self.input_dim )}
            sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
            x_uniform = sampler.random(n=int(n_samples))

            # These return NumPy arrays (unless isoprobabilistic_transform was fixed)
            x_doe_physical = isoprobabilistic_transform(x_uniform, uniform_marginals, self.physical_marginals)
            x_doe_norm = isoprobabilistic_transform(x_uniform, uniform_marginals, self.standard_marginals)
            
            # y_scaled is now a torch.float64 tensor from the fixed eval_lstate
            y_scaled = self.eval_lstate(x_doe_physical)


        return x_doe_norm, x_doe_physical, y_scaled
