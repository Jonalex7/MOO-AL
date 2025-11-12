import torch
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc

from utils.data import isoprobabilistic_transform

"""
Two-degree-of-freedom damped oscillator (Dubourg et al. 2013, Example 3)
Concave limit-state surface, 8D, lognormal marginals.
https://www.sciencedirect.com/science/article/pii/S0266892013000222?ref=pdf_download&fr=RR-2&rr=99cf46aa5992b726
Limit-state (Dubourg Eq. (26), OpenTURNS formulation):
    G = F_s - 3 k_s * sqrt( π S_0 / (4 ζ_s ω_s^3)
                            * [ (ζ_a ζ_s) / (ζ_p ζ_s (4 ζ_a^2 + θ^2) + γ ζ_a^2)
                                * ((ζ_p ω_p^3 + ζ_s ω_s^3) ω_p / (4 ζ_a ω_a^4)) ] )

with
    ω_p = sqrt(k_p / m_p)
    ω_s = sqrt(k_s / m_s)
    ω_a = (ω_p + ω_s)/2
    ζ_a = (ζ_p + ζ_s)/2
    γ   = m_s / m_p
    θ   = (ω_p - ω_s)/ω_a

Random variables (independent lognormal, mean & C.o.V.):
    m_p ~ Lognormal(μ=1.5,  COV=0.10)
    m_s ~ Lognormal(μ=0.01, COV=0.10)
    k_p ~ Lognormal(μ=1.0,  COV=0.20)
    k_s ~ Lognormal(μ=0.01, COV=0.20)
    ζ_p ~ Lognormal(μ=0.05, COV=0.40)
    ζ_s ~ Lognormal(μ=0.02, COV=0.50)
    F_s ~ Lognormal(μ=27.5, COV=0.10)  # you can change μFs if you want
    S_0 ~ Lognormal(μ=100,  COV=0.10)
"""

class g8d_two_dof_oscillator:
    def __init__(self, mu_Fs=15.0):
        self.input_dim = 8
        self.output_dim = 1

        # Reference probability of failure for μFs = 27.5 (Dubourg / OpenTURNS)
        # Not used in computations, just for comparison.
        self.target_pf = 0.0047598  # with MCS 1e7

        # Standard (normalized) space: all N(0,1)
        self.standard_marginals = {
            f'x{var+1}': [0.0, 1.0, 'norm'] for var in range(self.input_dim)
        }

        # Physical marginals: [mean, “std or COV”, distribution]
        # For lognormal we use [mean, COV, 'lognorm'] to match the paper & OpenTURNS.
        self.physical_marginals = {
            'x1': [1.5,   0.15, 'lognorm'],   # m_p
            'x2': [0.01,  0.001, 'lognorm'],   # m_s
            'x3': [1.0,   0.2, 'lognorm'],   # k_p
            'x4': [0.01,  0.002, 'lognorm'],   # k_s
            'x5': [0.05,  0.02, 'lognorm'],   # ζ_p
            'x6': [0.02,  0.01, 'lognorm'],   # ζ_s
            'x7': [mu_Fs, mu_Fs*0.1, 'lognorm'],   # F_s
            'x8': [100.0, 10, 'lognorm'],   # S_0
        }
        # Convention as in your other files:
        #   mean(or min), std(or max), marginal_distrib

    def eval_lstate(self, x):
        """
        Evaluate the limit-state in PHYSICAL space.
        x : array-like, shape (n_samples, 8) or (8,)
            Columns: [m_p, m_s, k_p, k_s, ζ_p, ζ_s, F_s, S_0]
        Returns: torch.Tensor of shape (n_samples,)
        """
        x = np.array(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        mp   = x[:, 0]
        ms   = x[:, 1]
        kp   = x[:, 2]
        ks   = x[:, 3]
        zetap = x[:, 4]
        zetas = x[:, 5]
        Fs   = x[:, 6]
        S0   = x[:, 7]

        # Derived mechanical quantities
        omega_p = np.sqrt(kp / mp)
        omega_s = np.sqrt(ks / ms)
        omega_a = 0.5 * (omega_p + omega_s)
        zeta_a  = 0.5 * (zetap + zetas)
        gamma   = ms / mp
        theta   = (omega_p - omega_s) / omega_a

        # Equation (26) / OpenTURNS implementation
        # inner1 = π S0 / (4 ζ_s ω_s^3)
        inner1 = np.pi * S0 / (4.0 * zetas * omega_s**3)

        # inner2 = (ζ_a ζ_s) / (ζ_p ζ_s (4 ζ_a^2 + θ^2) + γ ζ_a^2)
        denom2 = zetap * zetas * (4.0 * zeta_a**2 + theta**2) + gamma * zeta_a**2
        inner2 = (zeta_a * zetas) / denom2

        # inner3 = ((ζ_p ω_p^3 + ζ_s ω_s^3) ω_p) / (4 ζ_a ω_a^4)
        inner3 = ( (zetap * omega_p**3 + zetas * omega_s**3) * omega_p ) \
                 / (4.0 * zeta_a * omega_a**4)

        std_resp = np.sqrt(inner1 * inner2 * inner3)

        # Final limit-state:
        # G = F_s - 3 k_s * std_resp
        g = Fs - 3.0 * ks * std_resp

        return torch.as_tensor(g, dtype=torch.float32)

    def monte_carlo_estimate(self, n_samples):
        """
        Crude MC in standard normal space, then transform to physical space.
        Returns:
            Pf_ref (float), beta_ref (float), x_mc_physical, y_mc
        """
        n_mcs = int(n_samples)
        x_mc_norm = np.random.normal(0.0, 1.0, size=(n_mcs, self.input_dim))

        # Standard N(0,1) -> physical lognormal space
        x_mc_physical = isoprobabilistic_transform(
            x_mc_norm, self.standard_marginals, self.physical_marginals
        )
        y_mc = self.eval_lstate(x_mc_physical)
        Pf_ref = (y_mc < 0.0).float().mean()
        beta_ref = -norm.ppf(Pf_ref)

        return Pf_ref.item(), beta_ref, x_mc_physical, y_mc

    def get_doe(self, n_samples=10, method='lhs', random_state=None):
        """
        Initial design of experiments in *standard* space using LHS.
        Returns:
            x_doe_norm  : samples in standard N(0,1)^8
            x_doe_phys  : corresponding samples in physical space
            y_scaled    : g(x_doe_phys)
        """
        if random_state is None:
            random_state = np.random.RandomState()

        if method.lower() != 'lhs':
            raise NotImplementedError("Only 'lhs' is implemented here.")

        # LHS in unit hypercube
        uniform_marginals = {
            f'x{var+1}': [0.0, 1.0, 'uniform'] for var in range(self.input_dim)
        }
        sampler = qmc.LatinHypercube(d=self.input_dim, seed=random_state)
        x_uniform = sampler.random(n=int(n_samples))

        # Map to physical + standard normal space
        x_doe_physical = isoprobabilistic_transform(
            x_uniform, uniform_marginals, self.physical_marginals
        )
        x_doe_norm = isoprobabilistic_transform(
            x_uniform, uniform_marginals, self.standard_marginals
        )
        y_scaled = self.eval_lstate(x_doe_physical)

        return x_doe_norm, x_doe_physical, y_scaled
