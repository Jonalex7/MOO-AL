import torch
import numpy as np

class BatchActiveLearning():
    def __init__(self, n_active_samples):
        print('engine humming...')
        self.n_active_samples = n_active_samples
    
    def get_random(self, idx_pool):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        n_active_points = self.num_active_points
        random_idx_pool = np.random.choice(idx_pool, n_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()
        return random_idx_pool

    def get_u_function(self, mean_prediction, std_prediction, samples=None):
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples

        u_function = (mean_prediction.abs())/std_prediction
        _, u_idx = u_function.squeeze().topk(act_samples, largest=False)
        selected_indices = u_idx.tolist()
        return selected_indices

    def get_correlation(self, x_mc, model, mean_prediction, std_prediction, samples=None):
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples

        #firs sample evaluated with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, samples=1)

        for sample in range(act_samples-1):
            #Covariance computation
            det_cov = []
            for sample in range(len(x_mc)):
                x_assemble = x_mc[selected_indices + [sample]] 
                _, cov_assemble = model.predict(x_assemble, return_std=False, return_cov=True)
                det_ = np.linalg.det(cov_assemble)
                det_cov.append(det_)            
        
            det_cov = torch.tensor(det_cov)

            #evaluate U_function normalised with det_cov
            u_function = (mean_prediction.abs())/det_cov

            #avoid selected values
            u_function[selected_indices] = np.inf

            #adding selected indices
            _, u_min_idx = u_function.topk(1, largest=False)
            selected_indices.append(u_min_idx.item())

        return selected_indices