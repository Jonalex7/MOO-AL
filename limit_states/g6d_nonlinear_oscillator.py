import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

'''4.3. AK-MCS Example 3: dynamic response of a non-linear oscillator
It consists of a non-linear undamped single degree of freedom system'''

class g6d_nonlinear_oscillator():
    def __init__(self):
        self.input_dim = 6
        self.output_dim = 1
        self.target_pf = 0.0286178 # ref with MCS = 1e7
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim )}

        self.physical_marginals = {'x1': [1, 0.05, 'norm'], #m
                                'x2': [1, 0.1, 'norm'],     #c1
                                'x3': [0.1, 0.01, 'norm'],  #c2
                                'x4': [0.5, 0.05, 'norm'],  #r
                                'x5': [1, 0.2, 'norm'],     #F1
                                'x6': [1, 0.2, 'norm']}     #t1
        '''mean(or min), std(or max), marginal_distrib'''

    def eval_lstate(self, x):
        x = np.array(x, dtype='f')
        
        n_dim = len(x.shape)
        if n_dim == 1:
            x = np.array(x)[np.newaxis]
        
        m = x[:, 0]
        c1 = x[:, 1]
        c2 = x[:, 2]
        r = x[:, 3]
        f1 = x[:, 4]
        t1 = x[:, 5]

        w0 = np.sqrt((c1+c2)/m)
        g = 3*r - np.abs(((2*f1)/(m*w0**2)) * np.sin((t1 * w0)/2))

        return torch.tensor(g)    

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
            random_state = np.random.RandomState()

        if method == 'lhs':
            # Generates samples that are uniformly distributed within the unit hypercube [0,1]^d
            uniform_marginals = {f'x{var+1}': [0, 1.0, 'uniform'] for var in range(self.input_dim )}
            sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
            x_uniform = sampler.random(n=int(n_samples))

            # Converting samples from uniform to physical and standard space
            x_doe_physical = isoprobabilistic_transform(x_uniform, uniform_marginals, self.physical_marginals)
            x_doe_norm = isoprobabilistic_transform(x_uniform, uniform_marginals, self.standard_marginals)
            y_scaled = self.eval_lstate(x_doe_physical)
               
        #Sobol DoE
        '''if method == 'sobol':
            sampler = qmc.Sobol(d=self.input_dim, scramble=True)    #d=dimensionality
            sample = sampler.random_base2(m=exp_sobol)   #change m=exponent to increase the sample size
            l_bounds = [-2.0, -2.0]  #design domain for each variable in the physical space
            u_bounds = [2.0, 2.0]
            X_active = qmc.scale(sample, l_bounds, u_bounds)
            Y_active = self.eval_lstate(X_active)
            return X_active, Y_active'''

        return x_doe_norm, x_doe_physical, y_scaled