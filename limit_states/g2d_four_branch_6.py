import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform
"""     Schöbi et al. , ASCE J. Risk Unc. (2016)
        The four-branch function is a common benchmark in structural reliability analysis that describes
        the failure of a series system with four distinct component limit states. Its mathematical
        formulation reads (Waarts, 2000; Schueremans and Van Gemert, 2005a,b):
        where the input variables are modeled by two independent Gaussian random variables

        Parameters
        ----------
            x : numpy.array of float(s)
                Values of independent variables: columns are the different parameters/random variables (x1, x2,...xn) and rows are different parameter/random variables sets for different calls.

        Returns
        -------
            g_val_sys : numpy.array of float(s)
                Performance function value for the system.
            g_val_comp : numpy.array of float(s)
                Performance function value for each component.
            msg : str
                Accompanying diagnostic message, e.g. warning."""
class g2D_four_branch_6():
    def __init__(self):
        self.input_dim = 2
        self.output_dim = 1
        self.target_pf = 0.0044613 # ref with MCS = 1e7
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim )}

        self.physical_marginals = {'x1': [0, 1.0, 'norm'],
                                    'x2': [0, 1.0, 'norm']}
        '''mean(or min), std(or max), marginal_distrib'''

    def eval_lstate(self, x):
        g, g1, g2, g3, g4 = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        msg = 'Ok'
        x = np.array(x, dtype='f')

        n_dim = len(x.shape)
        if n_dim == 1:
            x = np.array(x)[np.newaxis]
        elif n_dim > 2:
            msg = 'Only available for 1D and 2D arrays.'
            return float('nan'), float('nan'), msg

        nrv_p = x.shape[1]
        if nrv_p != self.input_dim:
            msg = f'The number of random variables (x, columns) is expected to be {self.input_dim} but {nrv_p} is provided!'
        else:
            g1 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 - (x[:, 0] + x[:, 1])/np.sqrt(2)
            g2 = 3 + 0.1*(x[:, 0] - x[:, 1])**2 + (x[:, 0] + x[:, 1])/np.sqrt(2)
            g3 = (x[:, 0] - x[:, 1]) + (6/np.sqrt(2))
            g4 = (x[:, 1] - x[:, 0]) + (6/np.sqrt(2))
            g = np.amin(np.stack((g1, g2, g3, g4)), 0)

        g_val_sys = g
        #g_val_comp = np.stack((g1, g2, g3, g4))
        return torch.tensor(g_val_sys)
    
    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.normal(0, 1, size=(n_mcs, self.input_dim))
        x_mc_physical = isoprobabilistic_transform(x_mc_norm, self.standard_marginals, self.physical_marginals)
        y_mc = self.eval_lstate(x_mc_physical)
        Pf_ref = torch.sum(y_mc < 0) / n_mcs
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref.item(), B_ref, x_mc_physical, y_mc
    
    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()  # Default if no random state is passed
        
        # Generates samples that are uniformly distributed within the unit hypercube [0,1]^d
        uniform_marginals = {f'x{var+1}': [0, 1.0, 'uniform'] for var in range(self.input_dim )}
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_uniform = sampler.random(n=int(n_samples))

        # Converting samples from uniform to physical and standard space
        x_doe_physical = isoprobabilistic_transform(x_uniform, uniform_marginals, self.physical_marginals)
        x_doe_norm = isoprobabilistic_transform(x_uniform, uniform_marginals, self.standard_marginals)
        y_scaled = self.eval_lstate(x_doe_physical)

        return x_doe_norm, x_doe_physical, y_scaled