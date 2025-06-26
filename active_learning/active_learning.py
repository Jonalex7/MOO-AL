import torch
import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

class BatchActiveLearning():
    def __init__(self, n_active_samples):
        print('engine humming...')
        self.n_active_samples = n_active_samples

        # Define the strategy mapping as a class attribute
        self.al_strategy_mapping = {
            'u': (self.get_u_function, ['mean_prediction', 'std_prediction']),
            'corr_det': (self.get_corr_det, ['x_mc_pool', 'model', 'mean_prediction', 'std_prediction']),
            'corr_eigen': (self.get_corr_eigen, ['x_mc_pool', 'model', 'mean_prediction', 'std_prediction']),
            'corr_entropy': (self.get_corr_entropy, ['x_mc_pool', 'model', 'mean_prediction', 'std_prediction']),
            'corr_condvar': (self.get_corr_condvar, ['x_mc_pool', 'model', 'mean_prediction', 'std_prediction']),
            'random': (self.get_random, ['x_mc_pool']),
            'eff': (self.get_eff_function, ['mean_prediction', 'std_prediction']),
            'knee': (self.get_mo_function, ['mean_prediction', 'std_prediction']),
            'compromised': (self.get_mo_function, ['mean_prediction', 'std_prediction']),
        }

    def select_indices(self, al_strategy, **kwargs):
        if al_strategy in self.al_strategy_mapping:
            method, required_args = self.al_strategy_mapping[al_strategy]
            # Extract the required arguments from kwargs
            args = [kwargs[arg] for arg in required_args]
            
            # Specify the method parameter if needed for `get_mo_function`
            if al_strategy in ['knee', 'compromised']:
                method_name = 'knee' if al_strategy == 'knee' else 'compromised'
                return method(*args, method=method_name)
            else:
                return method(*args)
        else:
            raise ValueError(f"Unknown active learning strategy: {al_strategy}")

    def logistic_gamma(self, delta_P, delta_P0=0.1, k=20):
        gamma_max = 1.0
        gamma = gamma_max*(1 / (1 + np.exp(-k * (delta_P - delta_P0))))
        return gamma

    def get_mo_reliability(self, gamma, pareto_front):
        # Extract mean predictions and standard deviations
        mean_predictions = pareto_front[:, 0]
        std_predictions = pareto_front[:, 1]
        
        # Normalize the objectives to [0, 1]
        mean_min, mean_max = mean_predictions.min(), mean_predictions.max()
        std_min, std_max = std_predictions.min(), std_predictions.max()
        
        normalized_mean = (mean_predictions - mean_min) / (mean_max - mean_min)
        normalized_std = (std_predictions - std_min) / (std_max - std_min)
        
        # Calculate the scalar scores with the desired gamma mapping
        scores = (1 - gamma) * normalized_mean + gamma * normalized_std

        # Assign weights to samples
        weights = scores / scores.sum()
        arg_max = np.argmax(weights).item()
        # mo_reliability = pareto_front[arg_max]
        return arg_max
            
    def calculate_determinant(self, x_mc, model, sample, selected_indices):
        x_assemble = x_mc[selected_indices + [sample]]
        _, cov_assemble = model.predict(x_assemble, return_std=False, return_cov=True)
        return np.linalg.det(cov_assemble)
    
    def calculate_eigen(self, x_mc, model, sample, selected_indices):
        x_assemble = x_mc[selected_indices + [sample]]
        _, cov_assemble = model.predict(x_assemble, return_std=False, return_cov=True)
        eigen_values, _ = np.linalg.eig(cov_assemble)
        return np.sum(eigen_values)
    
    def differential_entropy(self, x_mc, model, sample, selected_indices):
        x_assemble = x_mc[selected_indices + [sample]]
        _, cov_assemble = model.predict(x_assemble, return_std=False, return_cov=True)
        # Differential entropy
        # Compute the dimensionality (number of samples)
        k = cov_assemble.shape[0]
        # Calculate the determinant of the covariance matrix
        det_cov = np.linalg.det(cov_assemble)
        if det_cov <= 0:
            det_cov = 1e-6

        differential_entropy = 0.5 * np.log((2 * np.pi * np.e)**k * det_cov)
        
        return max(differential_entropy, 0) 
    
    def get_random(self, x_mc, samples=None):
        # print('random')
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples

        random_idx_pool = np.random.choice(np.arange(len(x_mc)), act_samples, replace=False)
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

    def get_eff_function(self, mean_prediction, std_prediction, samples=None):

        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples
            
        eps = 2 * std_prediction

        # Compute each component using scipy's norm functions
        eff = (mean_prediction * (2 * norm.cdf(-mean_prediction / std_prediction) 
                    - norm.cdf(-(eps + mean_prediction) / std_prediction)
                    - norm.cdf((eps - mean_prediction) / std_prediction))
            - std_prediction * (2 * norm.pdf(-mean_prediction / std_prediction)
                        - norm.pdf(-(eps + mean_prediction) / std_prediction)
                        - norm.pdf((eps - mean_prediction) / std_prediction))
            + eps * (norm.cdf((eps - mean_prediction) / std_prediction)
                    - norm.cdf((-eps - mean_prediction) / std_prediction)))
        
        _, eff_idx = eff.squeeze().topk(act_samples)
        selected_indices = eff_idx.tolist()
        return selected_indices

    def get_mo_function(self, mean_prediction, std_prediction, method=None):

        _, _, _, original_knee_index, _, original_compromised_index, _ = self.compute_pareto_front(torch.abs(mean_prediction), std_prediction)

        if method == 'knee':
            selected_indices = original_knee_index.tolist()
        elif method == 'compromised':
            selected_indices = original_compromised_index.tolist()
        else:
            raise ValueError(f"Unknown MO pareto strategy: {method}")
        
        return [selected_indices]
    
    def compute_pareto_front(self, mean_pred: torch.Tensor, std_pred: torch.Tensor):
        # Negate mean_pred since we want to minimize it while using maximization logic
        objectives = torch.stack([-mean_pred, std_pred], dim=1)
        
        # Create a mask to track Pareto efficiency
        is_pareto_efficient = torch.ones(objectives.size(0), dtype=torch.bool)

        # Loop through all points and check if each is Pareto efficient
        for i, point in enumerate(objectives):
            # A point is Pareto efficient if no other point dominates it
            if is_pareto_efficient[i]:
                # Mark dominated points as not Pareto efficient
                is_dominated = torch.all(objectives <= point, dim=1) & torch.any(objectives < point, dim=1)
                is_pareto_efficient[is_dominated] = False

        # Extract the Pareto front points and their original indices
        pareto_front_indices = torch.nonzero(is_pareto_efficient, as_tuple=False).squeeze()
        pareto_front = objectives[is_pareto_efficient]
        
        # Sort the Pareto front by mean_pred (ascending) for knee point calculation
        sorted_indices = pareto_front[:, 0].argsort()
        pareto_front = pareto_front[sorted_indices]
        pareto_front_indices = pareto_front_indices[sorted_indices]

        # Calculate the knee point using the needle method
        knee_point, knee_index = self.calculate_knee_point(pareto_front)
        original_knee_index = pareto_front_indices[knee_index]

        # Calculate the compromised point from the ideal(utopian) point
        compromised_point, compromised_index, ideal_point = self.calculate_compromised_point(pareto_front)
        original_compromised_index = pareto_front_indices[compromised_index]
        
        return pareto_front, pareto_front_indices, knee_point, original_knee_index, compromised_point, original_compromised_index, ideal_point

    def calculate_knee_point(self, pareto_front: torch.Tensor):
        # Define the line between the first and last point in the Pareto front
        p1, p2 = pareto_front[0], pareto_front[-1]
        line_vec = p2 - p1
        line_vec /= torch.norm(line_vec)

        # Calculate the distance of each point on the Pareto front from the line
        distances = torch.zeros(pareto_front.size(0))
        for i, point in enumerate(pareto_front):
            point_vec = point - p1
            proj_len = torch.dot(point_vec, line_vec)
            proj_point = p1 + proj_len * line_vec
            distances[i] = torch.norm(point - proj_point)

        # The knee point is the point with the maximum distance from the line
        knee_index = torch.argmax(distances)
        knee_point = pareto_front[knee_index]
        
        return knee_point, knee_index

    def calculate_compromised_point(self, pareto_front: torch.Tensor):
        # Define the ideal point based on the objective directions
        ideal_point = torch.max(pareto_front, dim=0).values

        # Calculate the Euclidean distance from each point on the Pareto front to the ideal point
        distances = torch.norm(pareto_front - ideal_point, dim=1)
        
        # Find the index of the point with the minimum distance
        compromised_index = torch.argmin(distances)
        compromised_point = pareto_front[compromised_index]
        
        return compromised_point, compromised_index, ideal_point
    
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
    
    def get_corr_det(self, x_mc, model, mean_prediction, std_prediction, samples=None):
        # print('determinant')
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples
        #firs sample evaluated with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, samples=1)

        for _ in range(act_samples - 1):
            # Covariance computation
            # Use parallel processing to calculate the determinants for all samples
            det_cov = Parallel(n_jobs=-1)(delayed(self.calculate_determinant)(x_mc, model, sample, selected_indices) for sample in range(len(x_mc)))
        
            det_cov = torch.tensor(det_cov)

            #evaluate U_function normalised with det_cov
            u_function = (mean_prediction.abs())/det_cov

            #avoid selected values
            u_function[selected_indices] = np.inf

            #adding selected indices
            _, u_min_idx = u_function.topk(1, largest=False)
            selected_indices.append(u_min_idx.item())

        return selected_indices
    
    def get_corr_eigen(self, x_mc, model, mean_prediction, std_prediction, samples=None):
        # print('eigen')
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples
        #firs sample evaluated with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, samples=1)

        for _ in range(act_samples - 1):
            # Covariance computation
            # Use parallel processing to calculate the determinants for all samples
            sum_eigen = Parallel(n_jobs=-1)(delayed(self.calculate_eigen)(x_mc, model, sample, selected_indices) for sample in range(len(x_mc)))
        
            sum_eigen = torch.tensor(sum_eigen)

            #evaluate U_function normalised with det_cov
            u_function = (mean_prediction.abs())/sum_eigen

            #avoid selected values
            u_function[selected_indices] = np.inf

            #adding selected indices
            _, u_min_idx = u_function.topk(1, largest=False)
            selected_indices.append(u_min_idx.item())

        return selected_indices

    def get_corr_entropy(self, x_mc, model, mean_prediction, std_prediction, samples=None):
        # print('determinant')
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples
        #firs sample evaluated with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, samples=1)

        for _ in range(act_samples - 1):
            # Covariance computation
            # Use parallel processing to calculate the determinants for all samples
            det_cov = Parallel(n_jobs=-1)(delayed(self.differential_entropy)(x_mc, model, sample, selected_indices) for sample in range(len(x_mc)))
        
            det_cov = torch.tensor(det_cov)

            #evaluate U_function normalised with det_cov
            u_function = (mean_prediction.abs())/det_cov

            #avoid selected values
            u_function[selected_indices] = np.inf

            #adding selected indices
            _, u_min_idx = u_function.topk(1, largest=False)
            selected_indices.append(u_min_idx.item())

        return selected_indices

    def get_corr_condvar(self, x_mc, model, mean_prediction, std_prediction, block_size=300, samples=None):
        # print('determinant')
        if samples is None:
            act_samples = self.n_active_samples
        else:
            act_samples = samples
        # Define the block size to loop over the MC pool
        block_size = block_size
        # First sample obtained with U_function
        selected_indices = self.get_u_function(mean_prediction, std_prediction, samples=1)

        for _ in range(act_samples - 1):
            # Extract the selected samples
            selected_samples = x_mc[selected_indices]
            num_selected = len(selected_indices)

            u_min = torch.inf

            # Loop over x_mc_pool in blocks
            for i in range(0, len(x_mc), block_size):
                # Determine the end index of the block
                end_idx = min(i + block_size, len(x_mc))
                
                # Get the block indices
                block_indices = np.arange(i, end_idx)
                
                # Remove selected_indices from block_indices to avoid duplicates
                block_indices = np.setdiff1d(block_indices, selected_indices)
                
                # Get the block samples
                block_samples = x_mc[block_indices]
                
                # Prepend selected_samples to block_samples
                subset = np.vstack((selected_samples, block_samples))

                # Compute the covariance matrix for the subset
                _, cov_subset = model.predict(subset, return_std=False, return_cov=True)   

                # Partition the covariance matrix
                K_aa = cov_subset[:num_selected, :num_selected]
                K_ab = cov_subset[:num_selected, num_selected:]
                K_ba = cov_subset[num_selected:, :num_selected]
                K_bb = cov_subset[num_selected:, num_selected:]

                # Compute the inverse of K_aa
                # ----------------------------------------------------------------------------------------------------
                # Regularize if necessary to handle numerical issues
                jitter = 1e-6 * np.eye(K_aa.shape[0]) # Add jitter for numerical stability
                # Perform Cholesky decomposition of K_aa + jitter
                try:
                    L = np.linalg.cholesky(K_aa + jitter)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Cholesky decomposition failed. Consider increasing the jitter term.")
                # Solve K_aa @ X = K_ab
                Y = np.linalg.solve(L, K_ab)
                X = np.linalg.solve(L.T, Y)

                # Compute the diagonal of the conditional covariance
                diag_K_bb = np.diag(K_bb)
                cross_terms = np.sum(K_ba * X.T, axis=1)
                conditional_variances = diag_K_bb - cross_terms
                # ----------------------------------------------------------------------------------------------------
                # Evaluate U on the block
                u_block = mean_prediction[block_indices].abs() / conditional_variances
                u_min_block, u_min_idx = u_block.topk(1, largest=False)

                # Get the index with the min global U value
                if u_min_block < u_min:
                    selected_idx = block_indices[u_min_idx]
                    u_min = u_min_block

            selected_indices = selected_indices + [selected_idx]

        return selected_indices