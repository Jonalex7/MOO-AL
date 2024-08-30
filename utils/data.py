import numpy as np
from scipy.stats import norm, uniform, lognorm

def isoprob_transform (x_normalised, marginals):
    # input_dim = x_normalised.shape[1]
    # x_normalised = np.array(x_normalised)
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