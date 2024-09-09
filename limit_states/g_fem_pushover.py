import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
from joblib import Parallel, delayed

from utils.data import isoprob_transform

from limit_states.pushover_concentrated import PushoverConcetrated_mod

"""This example demonstrates how to perform a pushover (nonlinear static) analysis in OpenSees using a 2-story, 
1-bay steel moment resisting frame. In the first model, the nonlinear behavior is represented using the concentrated 
plasticity concept with rotational springs. The rotational behavior of the plastic regions in the model follows a bilinear 
hysteretic response based on the Modified Ibarra Krawinkler Deterioration Model (Ibarra et al. 2005, Lignos and Krawinkler 2009, 2010). 

For this example, all modes of cyclic deterioration are neglected. 
A leaning column carrying gravity loads is linked to the frame to simulate P-Delta effects

https://opensees.berkeley.edu/wiki/index.php/Pushover_Analysis_of_2-Story_Moment_Frame
"""
class g_pushover():
    def __init__(self):
        self.input_dim = 4
        self.output_dim = 1
        self.target = 0.002 # ref with MCS = 1e6 , pf=0.002007 B=2.87705

        self.marginals = {'x1': [14938.0, 14938.0 * 0.1, 'lognormal'], # Mybeam_mean = 10938.0   # yield moment at plastic hinge location (i.e., My of RBS section, if used)
                          'x2': [23350.0, 23350.0 * 0.1, 'lognormal'],  # yield moment of colum section
                          'x3': [38.5, 38.5 * 0.02, 'normal'],   # cross-sectional area column section W24x131 for Story 1 & 2 (elasticBeamColumn: 111, 121, 112, 122)
                          'x4': [45.0, 45.0 * 0.1, 'lognormal']}   # external load at node 12
                          
        '''mean(or min), std(or max), marginal_distrib'''

    def parallel_frame_eval (self, x):
        _ , _, max_baseshear_mc = PushoverConcetrated_mod(x)
        return max_baseshear_mc

    def eval_lstate(self, x):
        # Ref. Lateral loads
        lat2 = 16.255
        lat3 = 31.636
        ratio_ref = lat3/lat2

        max_baseshear_mc = Parallel(n_jobs=-1)(delayed(self.parallel_frame_eval)(x[sample][:-1]) for sample in range(len(x)))

        # Evaluation of frame external load
        ext_load = 2*x[:,-1] * (1+ratio_ref)
        g_pushover = max_baseshear_mc - ext_load
        return g_pushover
    
    def monte_carlo_estimate(self, n_samples):
        n_mcs = int(n_samples)
        x_mc_norm = np.random.uniform(0, 1, size=(int(n_mcs), self.input_dim))
        x_mc_scaled = isoprob_transform(x_mc_norm, self.marginals)
        y_mc = self.eval_lstate(x_mc_scaled)
        Pf_ref = np.sum(y_mc < 0) / n_mcs
        B_ref = - norm.ppf(Pf_ref)
        return Pf_ref, B_ref, x_mc_scaled, y_mc
    
    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        n_passive = int(n_samples)
        if random_state is None:
            random_state = np.random.RandomState()

        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_norm = sampler.random(n=n_passive)
        x_scaled = isoprob_transform(x_norm, self.marginals)
        y_scaled = self.eval_lstate(x_scaled)

        return x_norm, x_scaled, y_scaled
    
    #Sobol DoE
    '''def get_doe_points(self, exp_sobol):
        sampler = qmc.Sobol(d=self.input_dim, scramble=True)    #d=dimensionality
        sample = sampler.random_base2(m=exp_sobol)   #change m=exponent to increase the sample size
        l_bounds = [-2.0, -2.0]  #design domain for each variable in the physical space
        u_bounds = [2.0, 2.0]
        X_active = qmc.scale(sample, l_bounds, u_bounds)
        Y_active = self.eval_lstate(X_active)
        return X_active, Y_active'''