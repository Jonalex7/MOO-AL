from typing import List, Optional
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm
from utils.data import normalize_tensor

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