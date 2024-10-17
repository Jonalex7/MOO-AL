import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import qmc
from utils.data import isoprobabilistic_transform

"""
    Dubourg's Oscillator Function
    Parameters:
    X : numpy.ndarray
        N x M matrix including N samples of M stochastic parameters
        (X=[m_p, m_s, k_p, k_s, zeta_p, zeta_s, S_0, F_s])
    P : float or None
        Scalar parameter. By default: p = 3

    Returns:
    g : numpy.ndarray
        Column vector of length N including evaluations using Dubourg's Oscillator function
        """
class g8d_damped_oscillator():
    def __init__(self):
        self.input_dim = 8
        self.output_dim = 1
        self.target_pf = 0.004780999 # ref with MCS = 1e7
        self.standard_marginals = {f'x{var+1}': [0, 1.0, 'norm'] for var in range(self.input_dim )}
        self.physical_marginals={'x1': [1.5, 1.5*0.1, 'lognorm'], #1.5*[1 0.1]
                        'x2': [0.01, 0.01*0.1, 'lognorm'], #0.01*[1 0.1]
                        'x3': [1, 1*0.2, 'lognorm'],      #1*[1 0.2]
                        'x4': [0.01, 0.01*0.2, 'lognorm'], #0.01*[1 0.2]
                        'x5': [0.05, 0.05*0.4, 'lognorm'],    #0.05*[1 0.4]
                        'x6': [0.02, 0.02*0.5, 'lognorm'],       #0.02*[1 0.5]
                        'x7': [100, 100*0.1, 'lognorm'],    #100*[1 0.1]
                        'x8': [15, 15*0.1, 'lognorm']}    #15*[1 0.1]

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
    
    def eval_lstate(self, x, P=None):
        # Convert input to a torch tensor with float type
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Determine the number of dimensions
        n_dim = len(x.shape)

        if n_dim == 1:
            # Add a new axis if the input is 1D
            x = x.unsqueeze(0)
        
        # Assert that the input has 8 variables in the second dimension
        assert x.shape[1] == 8, 'Exactly 8 input variables needed'

        if P is None:
            P = 3
        
        # Random variables
        m_p = x[:, 0]  # primary mass
        m_s = x[:, 1]  # secondary mass
        k_p = x[:, 2]  # stiffness of the primary spring
        k_s = x[:, 3]  # stiffness of the secondary spring
        zeta_p = x[:, 4]  # damping ratio of the primary damper
        zeta_s = x[:, 5]  # damping ratio of the secondary damper
        s_0 = x[:, 6]  # Intensity of the white noise base acceleration (excitation)
        F_s = x[:, 7]  # Force capacity of the secondary spring

        # Abbreviations and ratios
        # natural frequency of the primary partial system / oscillator
        omega_p = torch.sqrt(k_p / m_p)

        # natural frequency of the secondary partial system / oscillator
        omega_s = torch.sqrt(k_s / m_s)

        # relative mass
        gamma = m_s / m_p

        # average natural frequency
        omega_a = (omega_p + omega_s) / 2

        # average damping ratio
        zeta_a = (zeta_p + zeta_s) / 2

        # tuning parameter
        theta = (omega_p - omega_s) / omega_a

        # Mean-square relative displacement
        mean_square_relative_displacement = (
            (torch.pi * s_0) / (4 * zeta_s * omega_s**3) *
            (zeta_a * zeta_s) / ((zeta_p * zeta_s * (4 * zeta_a**2 + theta**2)) + (gamma * zeta_a**2)) *
            (zeta_p * omega_p**3 + zeta_s * omega_s**3) * (omega_p / (4 * zeta_a * omega_a**4))
        )
        # Evaluation
        # performance function
        
        g = F_s - (P * k_s * torch.sqrt(mean_square_relative_displacement))

        return g