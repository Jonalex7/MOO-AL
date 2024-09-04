import torch
import numpy as np
from scipy.stats import norm, uniform, lognorm

def isoprob_transform (x_normalised, marginals):
    # Assuming source distribution of x_normalised between 0,1
    x_scaled = np.zeros_like(x_normalised)

    for margin in range (0, x_scaled.shape[1]):
        var = 'x' + str (margin + 1)
        if marginals[var][2] == 'normal':
            loc_ = marginals[var][0]
            scale_ = marginals[var][1]
            x_scaled[:, margin] = np.array(norm.ppf(x_normalised[:, margin], loc=loc_, scale=scale_))

        elif marginals[var][2] == 'uniform':
            loc_ = marginals[var][0]
            scale_ = marginals[var][1]
            x_scaled[:, margin] = np.array(uniform.ppf(x_normalised[:, margin], loc=loc_, scale=scale_-loc_))

        elif marginals[var][2] == 'lognormal':
            xlog_mean = np.array(marginals[var][0])
            xlog_std = np.array(marginals[var][1])
            # converting lognormal mean and std. dev.
            SigmaLogNormal = np.sqrt( np.log(1+(xlog_std/xlog_mean)**2))
            MeanLogNormal = np.log(xlog_mean) - SigmaLogNormal**2/2
            x_scaled[:, margin] = np.array(lognorm.ppf(x_normalised[:, margin], s=SigmaLogNormal, scale=xlog_mean)) 
    
    return x_scaled

def min_max_distance(tensor1, tensor2):
    # both tensors are of the same dimensionality
    assert tensor1.shape[1] == tensor2.shape[1]
    
    # Calculate the pairwise Euclidean distances
    distances = torch.cdist(tensor1, tensor2, p=2)  # p=2 computes the Euclidean distance
    # Flatten the distance matrix and filter out zero distances
    flattened_distances = distances.flatten()
    
    # Find the minimum non-zero distance
    min_distance = flattened_distances[flattened_distances > 0].min()
    # Find the maximum distance
    max_distance = distances.max()
    
    return min_distance, max_distance