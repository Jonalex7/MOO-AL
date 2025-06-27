from typing import List, Optional
import torch
from torch import Tensor
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
        elif method == 'compromise':
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
    

class AcquisitionStrategy:
    def __init__(
        self,
        acquisition_strategy: str,
        moo_method: Optional[str] = None,
    ):
        self.strategy = acquisition_strategy.lower().strip()
        # Only relevant when strategy == "mo"
        if self.strategy == "moo":
            if moo_method not in ("knee", "compromise"):
                raise ValueError("`moo_method` must be 'knee' or 'compromise'")
        self.moo_method = moo_method

    def get_indices(
        self,
        mean_prediction: Tensor,
        std_prediction: Tensor,
        n_samples: int = 1,
        skip_indices: Optional[List[int]] = None,
        constant: float = 2.0,
    ) -> List[int]:
        if self.strategy == "u":
            return self._u_function(mean_prediction, std_prediction, n_samples, skip_indices)
        elif self.strategy == "eff":
            return self._eff_function(mean_prediction, std_prediction, n_samples, skip_indices, constant)
        elif self.strategy == "moo":
            # moo_method was validated in __init__
            return self.get_moo(mean_prediction, std_prediction, self.moo_method)
        else:
            raise ValueError(f"Unknown acquisition strategy '{self.strategy}'")

    def _u_function(
        self,
        mean_prediction: Tensor,
        std_prediction: Tensor,
        n_samples: int,
        skip_indices: Optional[List[int]]
    ) -> List[int]:
        u = mean_prediction.abs() / std_prediction
        if skip_indices is not None:
            u[skip_indices] = float('inf')
        _, u_idx = u.squeeze().topk(n_samples, largest=False)
        return u_idx.tolist()

    def _eff_function(
        self,
        mean_prediction: Tensor,
        std_prediction: Tensor,
        n_samples: int,
        skip_indices: Optional[List[int]],
        constant: float = 2.0
    ) -> List[int]:
        eps = constant * std_prediction
        eff = (
            mean_prediction
            * (
                2 * norm.cdf(-mean_prediction / std_prediction)
                - norm.cdf(-(eps + mean_prediction) / std_prediction)
                - norm.cdf((eps - mean_prediction) / std_prediction)
            )
            - std_prediction
            * (
                2 * norm.pdf(-mean_prediction / std_prediction)
                - norm.pdf(-(eps + mean_prediction) / std_prediction)
                - norm.pdf((eps - mean_prediction) / std_prediction)
            )
            + eps
            * (
                norm.cdf((eps - mean_prediction) / std_prediction)
                - norm.cdf((-eps - mean_prediction) / std_prediction)
            )
        )
        if skip_indices is not None:
            eff[skip_indices] = float('-inf')
        _, eff_idx = eff.squeeze().topk(n_samples)
        return eff_idx.tolist()

    def get_moo(
        self,
        mean_prediction: Tensor,
        std_prediction: Tensor,
        method: Optional[str] = None
    ) -> List[int]:
        """
        Multi-objective selection via Pareto front.

        method: 'knee' or 'compromise'
        """
        # Compute absolute mean for minimization
        _, _, _, knee_idx, _, comp_idx, _ = self.compute_pareto_front(
            torch.abs(mean_prediction), std_prediction
        )
        if method == 'knee':
            return [int(knee_idx)]
        elif method == 'compromise':
            return [int(comp_idx)]
        else:
            raise ValueError(f"Unknown MO pareto strategy: {method}")

    def compute_pareto_front(
        self,
        mean_pred: Tensor,
        std_pred: Tensor
    ):
        # Negate mean_pred for minimization via maximization logic
        objectives = torch.stack([-mean_pred, std_pred], dim=1)
        is_pareto = torch.ones(objectives.size(0), dtype=torch.bool)
        for i, pt in enumerate(objectives):
            if is_pareto[i]:
                dominated = torch.all(objectives <= pt, dim=1) & torch.any(objectives < pt, dim=1)
                is_pareto[dominated] = False
        indices = torch.nonzero(is_pareto, as_tuple=False).squeeze()
        front = objectives[is_pareto]
        # Sort by first objective
        order = front[:,0].argsort()
        front = front[order]
        indices = indices[order]
        knee_pt, knee_idx = self.calculate_knee_point(front)
        comp_pt, comp_idx, ideal_pt = self.calculate_compromised_point(front)
        return front, indices, knee_pt, indices[knee_idx], comp_pt, indices[comp_idx], ideal_pt

    def calculate_knee_point(self, pareto_front: Tensor):
        p1, p2 = pareto_front[0], pareto_front[-1]
        line = p2 - p1
        line = line / torch.norm(line)
        dists = torch.zeros(pareto_front.size(0))
        for i, pt in enumerate(pareto_front):
            vec = pt - p1
            proj = p1 + torch.dot(vec, line) * line
            dists[i] = torch.norm(pt - proj)
        idx = torch.argmax(dists)
        return pareto_front[idx], idx

    def calculate_compromised_point(self, pareto_front: Tensor):
        ideal = torch.max(pareto_front, dim=0).values
        dists = torch.norm(pareto_front - ideal, dim=1)
        idx = torch.argmin(dists)
        return pareto_front[idx], idx, ideal