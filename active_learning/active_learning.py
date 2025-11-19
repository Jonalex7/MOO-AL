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
        eps_start: float = 1.0,     # start fully exploratory
        eps_end: float = 0.0,       # end fully exploitative
        eps_T: int = 100,             # number of calls to decay over
        portfolio_lambda: float = 2.0,   # Hedge balance (λ)
        portfolio_delta: float = 0.7,    # Memory factor (δ)
    ):
        self.strategy = acquisition_strategy.lower().strip()

        if self.strategy == "moo":
            if moo_method not in ("knee", "compromise", "reliability", "eps_greedy"):
                raise ValueError("`moo_method` must be 'knee', 'compromise', 'reliability', or 'eps_greedy'")
            self.moo_method = moo_method
            # Initialize reliability parameters only when using moo_reliability
            if self.moo_method == "reliability":
                self.N_it = N_it
                self.delta_P0 = delta_P0
                self.k_balance = k_balance
                self.Pf_prev = 0.0
                self.delta_Pf_buffer: List[float] = []
            # epsilon-greedy schedule state
            if self.moo_method == "eps_greedy":
                self.eps_start = float(eps_start)
                self.eps_end   = float(eps_end)
                self.eps_T     = int(eps_T)
                self._eps_t    = 0  # internal call counter

        # --- NEW: portfolio init ---
        if self.strategy == "portfolio":
            # order of arms (must match the call sequence below)
            self._arms: List[str] = ["u", "eff", "erf", "reif", "reif2"]
            self._K = len(self._arms)

            # Hedge state: total rewards G_i and probabilities p_i
            self._G = torch.zeros(self._K, dtype=torch.float64)                 # totals
            self._p = torch.full((self._K,), 1.0/self._K, dtype=torch.float64)  # probs

            self._lambda = float(portfolio_lambda)
            self._delta  = float(portfolio_delta)

            # tracking which arm selected each iteration
            self.portfolio_history: List[str] = []
            # counts per arm
            self.portfolio_counts = {a: 0 for a in self._arms}

    def get_indices(
        self,
        mean_prediction: Tensor, # Mean predictions from the model
        std_prediction: Tensor, # Standard deviations from the model
        input_candidates: Optional[Tensor] = None,
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

        # ---------- Non-MOO strategies (all handled uniformly) ----------
        # Portfolio: always select *one* sample via the portfolio step
        if self.strategy == "portfolio":
            if n_samples != 1:
                raise ValueError("Portfolio strategy currently supports n_samples=1 only.")

            idx = self._portfolio_step(
                mean_prediction=mean_prediction,
                std_prediction=std_prediction,
                input_candidates=input_candidates,
                skip_indices=skip_indices,
            )
            selected_indices = [idx]

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
                # EFF-based selection

        elif self.strategy == "erf":
            selected_indices = self._erf_function(mean_prediction, std_prediction, n_samples, skip_indices)
            if self.pareto_metrics:
                mean_pred_norm = normalize_tensor(torch.abs(mean_prediction))
                std_pred_norm = normalize_tensor(std_prediction)
                pareto, _, _, _, _, _, _ = self.compute_pareto_front(
                    mean_pred_norm, std_pred_norm)
                return pareto, selected_indices
            else:
                return selected_indices
        
        elif self.strategy == "reif":
            selected_indices = self._reif_function(mean_prediction, std_prediction, n_samples, skip_indices)
            if self.pareto_metrics:
                mean_pred_norm = normalize_tensor(torch.abs(mean_prediction))
                std_pred_norm = normalize_tensor(std_prediction)
                pareto, _, _, _, _, _, _ = self.compute_pareto_front(
                    mean_pred_norm, std_pred_norm)
                return pareto, selected_indices
            else:
                return selected_indices
        
        elif self.strategy == "reif2":
            selected_indices = self._reif2_function(mean_prediction, std_prediction, input_candidates, n_samples, skip_indices)
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

    def _erf_function(
        self,
        mean_prediction: torch.Tensor,
        std_prediction: torch.Tensor,
        n_samples: int,                         # number of samples to select
        skip_indices: Optional[List[int]] = None  # indices to skip in the pool
    ) -> List[int]:
        """
        Expected Risk Function (ERF) sampling criterion.
        Selects the indices corresponding to the largest ERF values.
        Reference: Yang et al. (2015) ALK-HRA with Expected Risk Function.
        """
        mu = mean_prediction.squeeze()
        sig = std_prediction.squeeze()

        sgn = torch.where(mu >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        z = mu / sig

        phi = torch.from_numpy(norm.pdf(z.numpy()))
        Phi_neg = torch.from_numpy(norm.cdf((-sgn * z).numpy()))

        erf_val = -sgn * mu * Phi_neg + sig * phi  # larger = higher expected risk

        if skip_indices is not None:
            erf_val[skip_indices] = float('-inf')

        _, idx = erf_val.topk(n_samples, largest=True)
        return idx.tolist()

    def _reif_function(
        self,
        mean_prediction: torch.Tensor,
        std_prediction: torch.Tensor,
        n_samples: int,
        skip_indices: Optional[List[int]] = None,
        w: float = 2.0,  # as suggested in the paper
    ) -> List[int]:
        """
        REIF selector (maximize): REIF = w*σ - E[|ĝ|],
        where E[|N(μ, σ²)|] = σ*sqrt(2/pi)*exp(-0.5*(μ/σ)^2) + μ*(1 - 2*Φ(μ/σ))
        Reference: Zhang, Wang & Sørensen (2019), RESS. REIF/REIF2. 
        """
        mu = mean_prediction.squeeze()
        sig = std_prediction.squeeze()

        # compute beta = mu/sig
        beta_np = (mu / sig)
        Phi_beta = torch.from_numpy(norm.cdf(beta_np.numpy()))

        # folded-normal expectation E|ĝ|
        term_var = (w - np.sqrt(2.0/np.pi) * torch.exp(-0.5 * (beta_np)**2))
        term_mean = mu * (1.0 - 2.0 * Phi_beta)

        reif = term_mean + sig * term_var # larger is better

        if skip_indices:
            reif[skip_indices] = float('-inf')

        k = min(n_samples, reif.numel() - (len(skip_indices) if skip_indices else 0))
        if k <= 0:
            return []
        _, idx = reif.topk(k, largest=True)
        return idx.tolist()
    
    def _reif2_function(
        self,
        mean_prediction: torch.Tensor,
        std_prediction: torch.Tensor,
        input_candidates: torch.Tensor,   # f_X(x) evaluated at each candidate (same shape as mu)
        n_samples: int,
        skip_indices: Optional[List[int]] = None,
        w: float = 2.0,
    ) -> List[int]:
        """
        REIF2 selector (maximize): REIF2 = REIF * f_X(x).
        Same REIF core as above, with multiplicative modulation by the input PDF.
        Reference: Zhang, Wang & Sørensen (2019), RESS. REIF/REIF2.
        """
        mu = mean_prediction.squeeze()
        sig = std_prediction.squeeze()
        fx = self.std_normal_pdf_product(input_candidates)

        # compute beta = mu/sig
        beta_np = (mu / sig)
        Phi_beta = torch.from_numpy(norm.cdf(beta_np.numpy()))

        # folded-normal expectation E|ĝ|
        term_var = (w - np.sqrt(2.0/np.pi) * torch.exp(-0.5 * (beta_np)**2))
        term_mean = mu * (1.0 - 2.0 * Phi_beta)

        reif = term_mean + sig * term_var # larger is better
        reif2 = reif * fx

        if skip_indices:
            reif2[skip_indices] = float('-inf')

        k = min(n_samples, reif2.numel() - (len(skip_indices) if skip_indices else 0))
        if k <= 0:
            return []
        _, idx = reif2.topk(k, largest=True)
        return idx.tolist()

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
        elif method == 'eps_greedy':
            pos_on_front = self.get_moo_eps_greedy(pareto_front)
            return pareto_front, [pareto_front_indices[pos_on_front].item()]
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
    
    def std_normal_pdf_product(self, input_candidates: np.ndarray) -> torch.Tensor:
        pdf = norm.pdf(input_candidates)                 # (N, D)
        pdf_joint = pdf.prod(axis=1)              # independent product
        return torch.from_numpy(pdf_joint)

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
    
    def _eps_value(self) -> float:
        """Linear decay epsilon in [eps_start -> eps_end] over eps_T calls."""
        if self.eps_T <= 0:
            return self.eps_end
        frac = min(1.0, self._eps_t / self.eps_T)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def get_moo_eps_greedy(self, pareto_front: torch.Tensor) -> int:
        """
        Deterministic epsilon-greedy along the sorted Pareto front.
        Maps epsilon to a position from 0 (explore) -> K-1 (exploit).
        """
        K = pareto_front.size(0)
        if K == 0:
            raise ValueError("Empty Pareto front.")
        eps = self._eps_value()              # 1.0 -> 0.0 over time
        pos = int(round((1.0 - eps) * (K - 1)))
        pos = max(0, min(K - 1, pos))        # clamp
        self._eps_t += 1                      # advance schedule after each use
        return pos

    def reset_eps_schedule(self):
        """Optional: call this if you want to restart from full exploration."""
        self._eps_t = 0

    def _portfolio_step(
        self,
        mean_prediction: torch.Tensor,
        std_prediction: torch.Tensor,
        input_candidates: Optional[torch.Tensor],  # needed for REIF2
        skip_indices: Optional[List[int]] = None
    ) -> int:
        """
        One Hedge update + selection:
        - Each arm proposes its best idx
        - Reward r_i = -|mu(best_i)|
        - Totals G_i <- δ G_i + r_i
        - p_i = softmax(λ * normalized(G))
        - Sample one arm by p_i and return its idx
        """
        if input_candidates is None:
            # Only REIF2 needs candidates; we still require it here to keep interface simple
            raise ValueError("`input_candidates` (N,D) is required (for REIF2 portfolio arm).")
        mu = mean_prediction.squeeze().to(torch.float64)
        sig = std_prediction.squeeze().to(torch.float64)

        # 1) Each arm proposes best index (reuses your existing functions)
        arm_best: List[int] = []
        arm_best.append(self._u_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._eff_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._erf_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._reif_function(mu, sig, n_samples=1, skip_indices=skip_indices)[0])
        arm_best.append(self._reif2_function(mu, sig, input_candidates, n_samples=1, skip_indices=skip_indices)[0])

        # 2) rewards r_i = -|mu(best_i)|
        mu_best = mu[torch.as_tensor(arm_best, dtype=torch.long)]
        rewards = -mu_best.abs().to(torch.float64)

        # 3) totals update with memory
        self._G = self._delta * self._G + rewards

        # 4) probabilities via softmax on normalized totals
        Gmax = float(self._G.max())
        Gmin = float(self._G.min())
        if Gmax == Gmin:
            self._p = torch.full((self._K,), 1.0 / self._K, dtype=torch.float64)
        else:
            q = (self._G - Gmax) / (Gmax - Gmin)   # ∈ [-1,0]
            logits = self._lambda * q
            m = float(logits.max())
            expv = torch.exp(logits - m)
            self._p = expv / expv.sum()

        # 5) sample one arm and return its proposed index
        arm_idx = int(np.random.choice(self._K, p=self._p.numpy()))
        chosen_idx = int(arm_best[arm_idx])

        # tracking
        chosen_arm = self._arms[arm_idx]
        self.portfolio_history.append(chosen_arm)
        self.portfolio_counts[chosen_arm] += 1

        return chosen_idx
