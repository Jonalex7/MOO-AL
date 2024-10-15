import torch
import numpy as np
from scipy.stats import norm, uniform, lognorm
import scipy.stats as stats
from scipy.optimize import fmin_l_bfgs_b

def isoprobabilistic_transform(x, source_marginals, target_marginals):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if len(x.shape) == 1:
        x = x.unsqueeze(0)
        
    transformed_x = torch.empty_like(x)
    
    for i, (source_params, target_params) in enumerate(zip(source_marginals.values(), target_marginals.values())):
        loc_source, scale_source, dist_source = source_params
        loc_target, scale_target, dist_target = target_params
        
        # Define source distribution
        if dist_source == 'lognorm':
            # Compute mu and sigma for source lognormal distribution
            mu_source = np.log(loc_source**2 / np.sqrt(loc_source**2 + scale_source**2))
            sigma_source = np.sqrt(np.log(1 + (scale_source / loc_source)**2))
            dist_source = stats.lognorm(s=sigma_source, scale=np.exp(mu_source))  # lognorm takes sigma and exp(mu)
        elif dist_source == 'uniform':
            dist_source = stats.uniform(loc=loc_source, scale=scale_source)
        else:
            dist_source = getattr(stats, dist_source)(loc=loc_source, scale=scale_source)

        # Define target distribution
        if dist_target == 'lognorm':
            # Compute mu and sigma for target lognormal distribution
            mu_target = np.log(loc_target**2 / np.sqrt(loc_target**2 + scale_target**2))
            sigma_target = np.sqrt(np.log(1 + (scale_target / loc_target)**2))
            dist_target = stats.lognorm(s=sigma_target, scale=np.exp(mu_target))
        elif dist_target == 'uniform':
            dist_target = stats.uniform(loc=loc_target, scale=scale_target)
        else:
            dist_target = getattr(stats, dist_target)(loc=loc_target, scale=scale_target)

        # Calculate the CDF of source samples
        cdf_source = dist_source.cdf(x[:, i])
        
        # Use the inverse CDF (PPF) of the target distribution to get transformed samples
        transformed_x[:, i] = torch.tensor(dist_target.ppf(cdf_source), dtype=torch.float32)
    
    if x.shape[0] == 1:
        return transformed_x.squeeze()
    else:
        return transformed_x

def custom_optimizer(obj_func, initial_theta, bounds):
    opt_res = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=1000)
    return opt_res[0], opt_res[1]