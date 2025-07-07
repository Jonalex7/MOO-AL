from typing import List, Optional
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm
from utils.data import normalize_tensor

class AcquisitionStrategy:
    """
    This class holds methods for acquisition functions such as
    U-function, EFF, and multi-objective Pareto-based selection (including reliability adaptation).
    """
    def __init__(
        self,
        acquisition_strategy: str, # 'u', 'eff', or 'moo'
        moo_method: Optional[str] = None, # 'knee', 'compromise', or 'reliability'
        N_it: int = 2, # Number of iterations to consider for moving average in reliability method
        delta_P0: float = 0.2, # (0,1) threshold of relative difference at which gamma=0.5
        k_balance: float = 40,  # Positive constant controlling how quickly gamma transition from 0 to 1
        pareto_metrics: bool = False, # If True, returns Pareto front and selected indices
    ):
        self.strategy = acquisition_strategy.lower().strip()

        if self.strategy == "moo":
            if moo_method not in ("knee", "compromise", "reliability"):
                raise ValueError("`moo_method` must be 'knee', 'compromise' or 'reliability'")
            self.moo_method = moo_method
        
        # Initialize reliability parameters only when using moo_reliability
        if self.strategy == "moo" and self.moo_method == "reliability":
            self.N_it = N_it
            self.delta_P0 = delta_P0
            self.k_balance = k_balance
            self.Pf_prev = 0.0
            self.delta_Pf_buffer: List[float] = []

        self.pareto_metrics = pareto_metrics

    def get_indices(
        self,
        mean_prediction: Tensor, # Mean predictions from the model
        std_prediction: Tensor, # Standard deviations from the model
        n_samples: int = 1, # Number of samples to select
        skip_indices: Optional[List[int]] = None, # Indices to skip in the pool
        constant: float = 2.0, # Constant for EFF function
        pf_estimate: Optional[float] = None # Current Pf estimate for reliability method (if applicable)
    ) -> List[int]:
        
        # Get indices based on the acquisition strategy
        # If pareto metrics are requested, compute and return the Pareto front
        # MOO-based selection
        if self.strategy == "moo":
            pareto, selected_indices = self.get_moo(mean_prediction, std_prediction, self.moo_method, pf_estimate=pf_estimate)
            if self.pareto_metrics:
                return pareto, selected_indices
            else:
                return selected_indices
        # U-based selection
        if self.strategy == "u":
            selected_indices = self._u_function(mean_prediction, std_prediction, n_samples, skip_indices)
            if self.pareto_metrics:
                mean_pred_norm = normalize_tensor(torch.abs(mean_prediction))
                std_pred_norm = normalize_tensor(std_prediction)
                pareto, _, _, _, _, _, _ = self.compute_pareto_front(
                    mean_pred_norm, std_pred_norm)
                return pareto, selected_indices
            else:
                return selected_indices
        # EFF-based selection
        elif self.strategy == "eff":
            selected_indices = self._eff_function(mean_prediction, std_prediction, n_samples, skip_indices, constant)
            if self.pareto_metrics:
                mean_pred_norm = normalize_tensor(torch.abs(mean_prediction))
                std_pred_norm = normalize_tensor(std_prediction)
                pareto, _, _, _, _, _, _ = self.compute_pareto_front(
                    mean_pred_norm, std_pred_norm)
                return pareto, selected_indices
            else:
                return selected_indices
        else:
            raise ValueError(f"Unknown acquisition strategy '{self.strategy}'")

    def _u_function(
        self,
        mean_prediction: Tensor,
        std_prediction: Tensor,
        n_samples: int, # Number of samples to select
        skip_indices: Optional[List[int]] # Indices to skip in the pool
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
        n_samples: int, # Number of samples to select
        skip_indices: Optional[List[int]], # Indices to skip in the pool
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
        method: Optional[str] = None,  # 'knee', 'compromise' or 'reliability'
        pf_estimate: Optional[Tensor] = None, # Current Pf estimate for reliability method (if applicable)
    ) -> List[int]:
        """
        Multi-objective selection via Pareto front.
        method: 'knee', 'compromise' or 'reliability'
        """
        # Compute the Pareto front
        mean_pred_norm = normalize_tensor(torch.abs(mean_prediction))
        std_pred_norm = normalize_tensor(std_prediction)
        pareto_front, pareto_front_indices, _, knee_idx, _, comp_idx, _ = self.compute_pareto_front(
            mean_pred_norm, std_pred_norm
        )
        # select the knee point, compromised point, or reliability point
        if method == 'knee':
            return pareto_front, [int(knee_idx)]
        elif method == 'compromise':
            return pareto_front, [int(comp_idx)]
        elif method == 'reliability':
            moo_pareto_index = self.get_moo_reliability(pareto_front=pareto_front, pf_estimate=pf_estimate)
            return pareto_front, [pareto_front_indices[moo_pareto_index].item()]
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
    
    def logistic_gamma(self, delta_P, delta_P0=0.2, k=40):
        gamma_max = 1.0
        gamma = gamma_max*(1 / (1 + np.exp(-k * (delta_P - delta_P0))))
        return gamma

    def get_moo_reliability(self, pareto_front, pf_estimate):
        # Checking Pf rel. difference to choose gamma behaviour
        Pf_current = pf_estimate

        # Calculate the relative difference from the previous Pf
        if self.Pf_prev != 0:
            delta_Pf = abs(Pf_current - self.Pf_prev) / self.Pf_prev
        else:
            delta_Pf = 1e2  # Handle division by zero

        # Update the buffer with the latest delta_Pf
        self.delta_Pf_buffer.append(delta_Pf)
        if len(self.delta_Pf_buffer) > self.N_it:
            self.delta_Pf_buffer.pop(0)  # Keep only the last N values

        delta_avg = float(np.mean(self.delta_Pf_buffer))
        # compute gamma and update Pf_prev
        gamma = self.logistic_gamma(delta_avg, delta_P0=self.delta_P0, k=self.k_balance)
        print(f'delta_pf_avg: {delta_avg:.3f}, gamma_log: {gamma:.3f} \n')
        # Update previous Pf for next iteration
        self.Pf_prev = Pf_current
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