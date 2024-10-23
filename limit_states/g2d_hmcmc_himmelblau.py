import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

'''Example 4: The Himmelblau Function Ref. (HMCMC High Dim, Prof. Kostas, pag. 16) 
which is particularly suitable for reliability examples with multiple separated failure domains. 
x1 and x2 are assumed to be independent standard normal random variables and 
the constant beta is used to define different levels of the failure probability. 
(beta = 95 for Ref. PF=1.65E-4)   (beta = 50 for Ref. PF=2.77E-7)'''

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
        x = np.array(x, dtype='f')
        
        n_dim = len(x.shape)
        if n_dim == 1:
            x = np.array(x)[np.newaxis]
            
        beta = 95
        term1 = (((0.75*x[:,0] - 0.5)**2 / 1.81) + ((0.75*x[:,1] - 0.5) / 1.81) - 11)**2
        term2 = (((0.75*x[:,0] - 1.0)/ 1.81) + ((0.75*x[:,1] - 0.5)**2 / 1.81) - 7)**2
        g = term1 + term2 - beta
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