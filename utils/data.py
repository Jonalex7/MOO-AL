import torch
import numpy as np
from scipy.stats import norm, uniform, lognorm
import scipy.stats as stats

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
            shape_source = scale_source / loc_source  # Shape parameter for lognorm
            dist_source = stats.lognorm(s=shape_source, loc=0, scale=loc_source)
        elif dist_source == 'uniform':
            dist_source = stats.uniform(loc=loc_source, scale=scale_source)
        else:
            dist_source = getattr(stats, dist_source)(loc=loc_source, scale=scale_source)

        # Define target distribution
        if dist_target == 'lognorm':
            shape_target = scale_target / loc_target  # Shape parameter for lognorm
            dist_target = stats.lognorm(s=shape_target, loc=0, scale=loc_target)
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

def min_max_distance(tensor1, tensor2):
    # both tensors are of the same dimensionality
    # assert tensor1.shape[1] == tensor2.shape[1]
    
    # Calculate the pairwise Euclidean distances
    distances = torch.cdist(tensor1, tensor2, p=2)  # p=2 computes the Euclidean distance
    # Flatten the distance matrix and filter out zero distances
    flattened_distances = distances.flatten()
    
    # Find the minimum non-zero distance
    min_distance = flattened_distances[flattened_distances > 1e-3].min()
    # Find the maximum distance
    max_distance = distances.max()
    
    return min_distance.item(), max_distance.item()